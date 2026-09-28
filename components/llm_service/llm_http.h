#ifndef LLM_HTTP_H_
#define LLM_HTTP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cJSON.h"
#include "esp_http_client.h"

namespace recording_service {
class RecordedClip;
}

// Shared HTTPS + JSON helpers for the provider backends.
namespace llm_service {
namespace http {

struct Response {
    int status_code = 0;
    std::string body;
    std::string error_code;  // transport-level failure; empty whenever an HTTP reply arrived
    std::string error_message;
    std::string upload_url;  // x-goog-upload-url header (Gemini resumable upload)
};

using Header = std::pair<std::string, std::string>;

// One-shot request with an in-memory body (leave `body` empty for a GET).
Response Perform(const std::string& url, esp_http_client_method_t method,
                 const std::vector<Header>& headers, const std::string& body, int timeout_ms,
                 const char* client_label);

// Streams a WAV (44-byte header + PCM16 chunks from the clip) wrapped in a caller-supplied
// prefix and suffix, so the same routine serves a raw upload or a multipart form part.
// Content-Length is prefix + wav + suffix.
Response PerformWavUpload(const std::string& url, const std::vector<Header>& headers,
                          const std::string& prefix, const recording_service::RecordedClip& clip,
                          const std::string& suffix, int timeout_ms, const char* client_label);

std::string TrimCopy(std::string value);
std::string TrimForLog(std::string value, size_t max_len = 96);
std::string JsonStringField(cJSON* root, const char* key);
std::string JsonNestedStringField(cJSON* root, const char* first, const char* second);
// Serializes `root` (unformatted) without taking ownership.
std::string JsonToString(cJSON* root);
std::array<uint8_t, 44> BuildWavHeaderPcm16Mono(size_t sample_count, uint32_t sample_rate_hz);

}  // namespace http
}  // namespace llm_service

#endif  // LLM_HTTP_H_
