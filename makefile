# ============================================================
# 1. Configurações Gerais
# ============================================================

OMP ?= 1
CUDA ?= 0

ifneq ($(OMP), 0)
export OMP_NUM_THREADS=$(OMP)
endif

# Configuração do compilador baseada na opção CUDA
ifeq ($(CUDA), 1)
CXX      = nvcc
CXXFLAGS = --extended-lambda -std=c++20 -DPINEAPPLE_CUDA_ENABLED -Xcompiler -fopenmp -Iinclude -Wno-deprecated-gpu-targets
LDFLAGS  = -lcudart
$(info Building with nvcc (CUDA enabled), using OpenMP)
else
CXX      = g++
CXXFLAGS = -fopenmp -std=c++20 -Wall -Iinclude
LDFLAGS  =
$(info Building with g++ (no CUDA), using OpenMP)
endif

# ============================================================
# 2. Diretórios
# ============================================================
SRC_DIR     := src
LIB_DIR     := lib
OBJ_DIR     := $(LIB_DIR)/obj
BIN_DIR     := $(LIB_DIR)/bin
INCLUDE_DIR := inc

# ============================================================
# 3. Fontes e Objetos
# ============================================================ 
ifeq ($(CUDA), 1)
# Incluir arquivos .cpp e .cu quando CUDA estiver habilitado
CPP_FILES  := ./main.cpp $(shell find $(SRC_DIR) -name "*.cpp")
CU_FILES   := $(shell find $(SRC_DIR) -name "*.cu")
SRC_FILES  := $(CPP_FILES) $(CU_FILES)
OBJ_FILES  := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES)) \
              $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CU_FILES))
else
# Apenas arquivos .cpp quando CUDA estiver desabilitado
SRC_FILES  := ./main.cpp $(shell find $(SRC_DIR) -name "*.cpp")
OBJ_FILES  := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
endif

OBJ_DIRS   := $(sort $(dir $(OBJ_FILES)))
BIN_FILE   := $(BIN_DIR)/program

# ============================================================
# 4. Regras
# ============================================================
.PHONY: all build run clean rerun valgrind time help

all: build

help:
	@echo "Available targets:"
	@echo "  build     - Compile the project"
	@echo "  run       - Build and run the project"
	@echo "  clean     - Remove all build files"
	@echo "  rerun     - Clean, build and run"
	@echo "  time      - Build and run with time measurement"
	@echo "  valgrind  - Build and run with valgrind"
	@echo ""
	@echo "Options:"
	@echo "  CUDA=1    - Enable CUDA compilation with nvcc"
	@echo "  OMP=N     - Set number of OpenMP threads (default: 1)"
	@echo ""
	@echo "Examples:"
	@echo "  make CUDA=1          - Build with CUDA support"
	@echo "  make CUDA=1 run      - Build with CUDA and run"
	@echo "  make OMP=4 run       - Build with 4 OpenMP threads and run"

# Linka todos os objetos
build: $(BIN_FILE)

$(BIN_FILE): $(OBJ_FILES) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) -o $@ $(LDFLAGS)

# Compila .cpp → .o (fontes na raiz)
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compila .cpp → .o (fontes dentro de SRC_DIR)
$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compila .cu → .o (fontes CUDA dentro de SRC_DIR)
$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Criação de diretórios
$(BIN_DIR) $(OBJ_DIR):
	@mkdir -p $@

$(OBJ_DIRS):
	@mkdir -p $@

# Executa
run: build
	./$(BIN_FILE)

# Remove tudo
clean:
	rm -rf $(LIB_DIR)

# Roda com medição de tempo
time: build
	time -f %E ./$(BIN_FILE)

# Rerun completo
rerun: clean build run

# Valgrind
valgrind: build
	valgrind --leak-check=full --track-origins=yes ./$(BIN_FILE)
