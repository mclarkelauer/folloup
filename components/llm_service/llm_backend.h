#ifndef LLM_BACKEND_H_
#define LLM_BACKEND_H_

#include <string>

#include "llm_service.h"

namespace recording_service {
class RecordedClip;
}

// Private provider interface. Each backend is a stateless HTTP adapter for one provider; the
// service owns settings, readiness state, threading, and the portal routes.
namespace llm_service {
namespace backend {

struct AuthResult {
    bool success = false;
    int http_status = 0;
    std::string model_resource_name;
    std::string model_display_name;
    std::string error_code;
    std::string error_message;
};

class Backend {
 public:
    virtual ~Backend() = default;
    virtual Provider provider() const = 0;
    virtual std::string TextModel() const = 0;
    virtual std::string TranscriptionModel() const = 0;
    // Cheap credential probe (a model GET), run once per key / network change.
    virtual AuthResult Authenticate(const std::string& api_key) = 0;
    virtual TextResult GenerateText(const std::string& api_key, const std::string& prompt) = 0;
    virtual TokenCountResult CountTokens(const std::string& api_key,
                                         const std::string& prompt) = 0;
    virtual TranscriptionResult Transcribe(const std::string& api_key,
                                           const recording_service::RecordedClip& clip) = 0;
};

Backend& GeminiBackend();
Backend& MuseBackend();
Backend& BackendFor(Provider provider);

}  // namespace backend
}  // namespace llm_service

#endif  // LLM_BACKEND_H_
