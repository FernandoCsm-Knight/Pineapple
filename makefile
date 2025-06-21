# ============================================================
# 1. Configurações Gerais
# ============================================================

OMP ?= 1

ifneq ($(OMP), 0)
export OMP_NUM_THREADS=$(OMP)
endif

CXX      = g++
CXXFLAGS = -fopenmp -std=c++20 -Wall -Iinclude
LDFLAGS  =
$(info Building with g++ (no MPI), using OpenMP)

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
SRC_FILES  := ./main.cpp $(wildcard $(SRC_DIR)/**/*.cpp)
OBJ_FILES  := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
OBJ_DIRS   := $(sort $(dir $(OBJ_FILES)))
BIN_FILE   := $(BIN_DIR)/program

# ============================================================
# 4. Regras
# ============================================================
.PHONY: all build run clean rerun valgrind time

all: build

# Linka todos os objetos
build: $(BIN_FILE)

$(BIN_FILE): $(OBJ_FILES) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) -o $@ $(LDFLAGS)

# Compila .cpp → .o (fontes na raiz)
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compila .cpp → .o (fontes dentro de SRC_DIR)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIRS)
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
