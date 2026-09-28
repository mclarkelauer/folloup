idf_component_register(
    SRCS "llm_service.cpp" "llm_http.cpp" "gemini_backend.cpp" "muse_backend.cpp"
    INCLUDE_DIRS "include"
    PRIV_INCLUDE_DIRS "."
    REQUIRES esp_http_client esp_http_server esp-tls esp_timer json mbedtls nvs_flash recording_service task_config
)

target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)
