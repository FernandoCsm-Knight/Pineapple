#ifndef CUDA_DETECTOR_HPP
#define CUDA_DETECTOR_HPP

#include <vector>
#include <string>

namespace pineapple {

    /**
     * @brief Estrutura para armazenar informações de um dispositivo CUDA
     */
    struct CudaDeviceInfo {
        int device_id;
        std::string name;
        size_t total_memory;
        size_t free_memory;
        int compute_capability_major;
        int compute_capability_minor;
        int multiprocessor_count;
        int max_threads_per_block;
        int max_threads_per_multiprocessor;
        int warp_size;
        size_t shared_memory_per_block;
        size_t constant_memory;
        int max_grid_size_x;
        int max_grid_size_y;
        int max_grid_size_z;
        int max_block_dim_x;
        int max_block_dim_y;
        int max_block_dim_z;
        bool unified_addressing;
        bool can_map_host_memory;
    };

    /**
     * @brief Verifica se CUDA está disponível no sistema
     * @return true se CUDA estiver disponível, false caso contrário
     */
    bool is_cuda_available();

    /**
     * @brief Obtém o número de dispositivos CUDA disponíveis
     * @return Número de dispositivos CUDA disponíveis
     */
    int cuda_device_count();

    /**
     * @brief Obtém informações detalhadas de um dispositivo CUDA específico
     * @param device_id ID do dispositivo (0-based)
     * @return Estrutura CudaDeviceInfo com informações do dispositivo
     * @throws std::runtime_error se o device_id for inválido ou CUDA não estiver disponível
     */
    CudaDeviceInfo cuda_device_info(int device_id);

    /**
     * @brief Lista todos os dispositivos CUDA disponíveis
     * @return Vector contendo informações de todos os dispositivos CUDA
     */
    std::vector<CudaDeviceInfo> list_cuda_devices();

    /**
     * @brief Obtém o ID do dispositivo CUDA atualmente ativo
     * @return ID do dispositivo ativo, ou -1 se CUDA não estiver disponível
     */
    int current_cuda_device();

    /**
     * @brief Define o dispositivo CUDA ativo
     * @param device_id ID do dispositivo a ser definido como ativo
     * @return true se bem-sucedido, false caso contrário
     */
    bool set_cuda_device(int device_id);

    /**
     * @brief Verifica se um dispositivo CUDA específico está disponível
     * @param device_id ID do dispositivo a ser verificado
     * @return true se o dispositivo estiver disponível, false caso contrário
     */
    bool is_cuda_device_available(int device_id);

    /**
     * @brief Obtém informações sobre a memória de um dispositivo CUDA
     * @param device_id ID do dispositivo (opcional, usa o dispositivo atual se não especificado)
     * @return Par com (memória livre, memória total) em bytes
     * @throws std::runtime_error se o dispositivo for inválido ou CUDA não estiver disponível
     */
    std::pair<size_t, size_t> cuda_memory_info(int device_id = -1);

    /**
     * @brief Verifica se o dispositivo suporta uma determinada compute capability
     * @param device_id ID do dispositivo
     * @param major_version Versão maior da compute capability
     * @param minor_version Versão menor da compute capability
     * @return true se o dispositivo suportar a compute capability especificada ou superior
     */
    bool supports_compute_capability(int device_id, int major_version, int minor_version);

    /**
     * @brief Obtém a versão do driver CUDA
     * @return String com a versão do driver CUDA, ou string vazia se não disponível
     */
    std::string cuda_driver_version();

    /**
     * @brief Obtém a versão do runtime CUDA
     * @return String com a versão do runtime CUDA, ou string vazia se não disponível
     */
    std::string cuda_runtime_version();

} // namespace pineapple

#endif // CUDA_DETECTOR_HPP
