"""模型后端适配层。

runtime 只关心一件事：给我一个 prompt，我拿回一段文本。
不同 provider 在 HTTP 接口、响应结构、是否支持 prompt cache 上都有差异，
这些差异都在这里被抹平成统一的 complete() 接口。
"""

import json
import time
from http.client import RemoteDisconnected
import urllib.error
import urllib.request


def _retry_delay_seconds(attempt):
    """统一计算模型层重试退避时间。"""
    return 0.5 * (attempt + 1)


class FakeModelClient:
    """测试替身模型，按预设顺序吐出固定响应。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        """记录 prompt，并返回下一条预设输出。"""
        self.prompts.append(prompt)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)


class OllamaModelClient:
    """Ollama `/api/generate` 的最薄适配层。"""

    def __init__(self, model, host, temperature, top_p, timeout):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        # Ollama 当前不支持我们这里接入的 prompt cache 语义，
        # 所以 runtime 传下来的缓存参数会被忽略。
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")


def _normalize_versioned_base_url(base_url):
    """保证兼容后端 base URL 以 `/v1` 结尾。"""
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _extract_openai_text(data):
    """尽量兼容不同 OpenAI-compatible 响应形状提取文本。"""
    #如果传进来的 data 里有 output_text 字段,直接返回它的值,这是最简单直接的情况
    if data.get("output_text"):
        return data["output_text"]

    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if text:
                    return text

    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return text

    return ""

def _extract_openai_response_from_sse(body_text):
    """从 SSE 文本流里同时提取文本和最后一份响应对象。"""
    last_response = None
    deltas = []
    #将文本切成一个多行字符串,遍历每一行
    for line in body_text.splitlines():
        line = line.strip()
        #SSE 流里每一行都以字段名开头（data:、event:、id: 等），只有 data: 行才携带实际数据。这行的意思是：跳过所有非 data: 的行
        if not line.startswith("data:"):
            continue
        #选中这一行之后,应该截取data:之后到结尾的内容,并去掉首尾空白,得到的 payload 就是这一行携带的实际数据了
        payload = line[len("data:"):].strip()
        #如果 payload 为空或者 payload 是 [DONE] 表示这是最后一行,跳过
        if not payload or payload == "[DONE]":
            continue
        #尝试将这行转为json,如果解析失败,则跳过,防止错误格式的行导致整个解析崩溃
        try:
            #这个event解析出来就是一个json,也就是dict
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if event.get("error"):
            return "", {"error": event.get("error")}

        '''接下来是正式的解析过程'''
        #获取 response 字段,如果它是一个 dict 就暂存到 last_response 里,并尝试从里边提取文本
        response = event.get("response")
        if isinstance(response, dict):
            if response.get("error"):
                return "", {"error": response.get("error")}
            last_response = response
            if event.get("type") == "response.completed":
                text = _extract_openai_text(response)
                if text:
                    return text, response
        #这个可以将event这行的类型拿出来,判断如果是 delta 就把 delta 累积到 deltas 里,如果是 done 就直接返回 done 里边的文本,如果是其他类型就尝试从这一行里提取文本,如果提取到了就直接返回
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        #如果这一行的类型是 response.output_text.done,就直接从这一行里提取文本并返回,同时把最后的响应对象也返回,供后续提取 usage 和 cache 细节用 
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text, last_response or {}
        #如果是其他类型的行,就尝试从这一行里提取文本,如果提取到了就直接返回
        else:
            text = _extract_openai_text(event)
            if text:
                return text, event
    if deltas:
        return "".join(deltas), last_response or {}
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response), last_response
    return "", {}


def _extract_usage_cache_details(data):
    # 把不同 OpenAI-compatible 返回里的 usage 字段整理成统一结构，
    # 让 runtime/trace/report 不需要关心 provider 细节。
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_hit": cached_tokens > 0,
    }


class OpenAICompatibleModelClient:
    """兼容 OpenAI `/responses` 接口的模型客户端。"""

    def __init__(self, model, base_url, api_key, temperature, timeout):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        # 当前只在明确支持 prompt cache 语义的后端上启用这条链路，
        # 避免对不支持的后端传一个“看起来统一、其实没意义”的伪参数。
        #判断模型后端的 base_url 是否包含 "openai.com" 或 "right.codes"，只有这两个后端支持 OpenAI Responses API 的 prompt cache 协议。
        self.supports_prompt_cache = any(host in self.base_url for host in ("openai.com", "right.codes"))
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        """向 OpenAI-compatible `/responses` 接口发起一次模型调用。

        为什么存在：
        runtime 不应该知道 HTTP 细节、SSE 细节、usage 字段长什么样，
        更不应该自己去判断 prompt cache 参数要不要带。这个函数把这些后端
        细节都包起来，对上层暴露统一的 `complete()` 行为。

        输入 / 输出：
        - 输入：完整 prompt、最大输出 token，以及可选的 prompt cache 参数
        - 输出：模型最终文本；同时把 usage / cached_tokens 等元数据写进
          `self.last_completion_metadata`

        在 agent 链路里的位置：
        它位于 `Hepilot.ask()` 的模型调用阶段，是稳定前缀缓存复用链路真正
        落到 provider API 的地方。
        """
        #每次回答前先将上一次回答细节清空,准备这次存入
        self.last_completion_metadata = {}
        #构建payload数据负载,规定格式
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        # runtime 传入的是“稳定前缀”的签名，而不是整段 prompt 的签名。
        # 这样缓存复用针对的是稳定段，不会因为动态 history 每轮变化而失效。
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention
        #构建http头
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        #构造请求体
        request = urllib.request.Request(
            self.base_url + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        # 模型层只对“响应形态异常 / 瞬时后端故障”做有限重试；
        # 这样 runtime 不需要接管 provider 级网络抖动恢复。
        attempts = 3
        for attempt in range(attempts):
            try:
                # 通过 HTTP POST 把前面拼好的 request 发到模型后端，timeout=self.timeout 设置超时时间，with 保证响应对象用完后自动关闭连接。
                # response 就是后端返回的 HTTP 响应对象，接下来读 body、取 headers 都在这个作用域里进行。
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    #decode把字节按 UTF-8 解码成文本
                    #因为模型 API 返回的是 JSON 格式的文本，所以这个字符串长这样：
                    #{"output": [{"content": [{"text": "模型回复的内容..."}]}], "usage": {...}}
                    #后面再用 json.loads(body_text) 把它转成 Python dict。
                    body_text = response.read().decode("utf-8")
                    headers = getattr(response, "headers", {}) or {}
                    content_type = headers.get("Content-Type", "")
            except urllib.error.HTTPError as exc:  # HTTP 错误（4xx/5xx）
                body = exc.read().decode("utf-8", errors="replace")  # 读出后端返回的错误体文本
                if exc.code >= 500 and attempt < attempts - 1:  # 仅 5xx 服务端错误才值得重试
                    time.sleep(_retry_delay_seconds(attempt))  # 线性退避：0.5s → 1.0s → 1.5s
                    continue  # 重试下一轮
                raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc  # 4xx 或重试耗尽，直接报错
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                if attempt < attempts - 1:
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise RuntimeError(
                    "Could not reach the OpenAI-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc
            # 请求成功后，继续在同一次 attempt 里做响应解析；
            # 这样“请求阶段”和“响应阶段”的重试都统一落在模型层内部。
            if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
                text, response_data = _extract_openai_response_from_sse(body_text)
                if isinstance(response_data, dict) and response_data.get("error"):
                    raise RuntimeError(f"OpenAI-compatible error: {response_data['error']}")
                if isinstance(response_data, dict) and response_data:
                    # 这些元数据会一路传回 runtime，进入 trace 和 report，
                    # 用来观察 prompt cache 是否真的命中。
                    self.last_completion_metadata = {
                        "prompt_cache_supported": self.supports_prompt_cache,
                        "prompt_cache_key": prompt_cache_key,
                        "prompt_cache_retention": prompt_cache_retention,
                        **_extract_usage_cache_details(response_data),
                    }
                if text:
                    return text
                # SSE 拿到了响应但没能抽出正文，更像一次异常响应，优先在模型层内部补重试。
                if attempt < attempts - 1:
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise RuntimeError("OpenAI-compatible error: could not extract text from event stream response")

            try:
                data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                # 非法 JSON 说明这次响应体不稳定，先在模型层内部补重试，再决定是否失败。
                if attempt < attempts - 1:
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise RuntimeError(
                    "OpenAI-compatible error: backend returned non-JSON content that could not be parsed"
                ) from exc
            if data.get("error"):
                raise RuntimeError(f"OpenAI-compatible error: {data['error']}")
            self.last_completion_metadata = {
                "prompt_cache_supported": self.supports_prompt_cache,
                "prompt_cache_key": prompt_cache_key,
                "prompt_cache_retention": prompt_cache_retention,
                **_extract_usage_cache_details(data),
            }
            text = _extract_openai_text(data)
            if text:
                return text
            # 正常拿到 JSON 但正文为空，仍然按异常响应处理，避免把空回复交给 runtime。
            if attempt < attempts - 1:
                time.sleep(_retry_delay_seconds(attempt))
                continue
            raise RuntimeError("OpenAI-compatible error: could not extract text from response")


def _extract_anthropic_text(data):
    """从 Anthropic-compatible `content` 数组里抽取纯文本。"""
    for item in data.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                return text
    return ""


class AnthropicCompatibleModelClient:
    """兼容 Anthropic `/messages` 接口的模型客户端。"""

    def __init__(self, model, base_url, api_key, temperature, timeout):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        """向 Anthropic-compatible 后端发起一次完整文本请求。"""
        # 为了保持统一接口，runtime 仍然会传缓存参数进来；
        # 这里只是显式丢弃，因为当前 Anthropic-compatible 路径没有接缓存复用。
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        request = urllib.request.Request(
            self.base_url + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body_text = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code >= 500 and attempt < attempts - 1:
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise RuntimeError(f"Anthropic-compatible request failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, RemoteDisconnected) as exc:
                if attempt < attempts - 1:
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise RuntimeError(
                    "Could not reach the Anthropic-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from exc
            try:
                data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                # 非法 JSON 更像协议层抖动，先让模型层做一次有限补救。
                if attempt < attempts - 1:
                    time.sleep(_retry_delay_seconds(attempt))
                    continue
                raise RuntimeError(
                    "Anthropic-compatible error: backend returned non-JSON content that could not be parsed"
                ) from exc
            if data.get("error"):
                raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
            text = _extract_anthropic_text(data)
            if text:
                return text
            # 已经拿到 JSON 但抽不出正文，说明响应形态异常，先在模型层内部补重试。
            if attempt < attempts - 1:
                time.sleep(_retry_delay_seconds(attempt))
                continue
            raise RuntimeError("Anthropic-compatible error: could not extract text from response")
