export

# Disable built-in suffix and implicit pattern rules (for software builds).
# This makes starting with a very large number of GT lines much faster.
MAKEFLAGS += -r

## Make sure that sort always uses the same sort order.
LC_ALL := C

SHELL := /bin/bash
LOCAL := $(PWD)/usr
PATH := $(LOCAL)/bin:$(PATH)

# Path to the .traineddata directory with traineddata suitable for training
# (for example from tesseract-ocr/tessdata_best). Default: $(LOCAL)/share/tessdata
TESSDATA =  $(LOCAL)/share/tessdata

# Name of the model to be built. Default: $(MODEL_NAME)
MODEL_NAME = foo

# Data directory for output files, proto model, start model, etc. Default: $(DATA_DIR)
DATA_DIR = data

# Data directory for langdata (downloaded from Tesseract langdata repo). Default: $(LANGDATA_DIR)
LANGDATA_DIR = $(DATA_DIR)/langdata

# Output directory for generated files. Default: $(OUTPUT_DIR)
OUTPUT_DIR = $(DATA_DIR)/$(MODEL_NAME)

# Ground truth directory. Default: $(GROUND_TRUTH_DIR)
GROUND_TRUTH_DIR := $(OUTPUT_DIR)-ground-truth

# Optional Wordlist file for Dictionary dawg. Default: $(WORDLIST_FILE)
WORDLIST_FILE := $(OUTPUT_DIR)/$(MODEL_NAME).wordlist

# Optional Numbers file for number patterns dawg. Default: $(NUMBERS_FILE)
NUMBERS_FILE := $(OUTPUT_DIR)/$(MODEL_NAME).numbers

# Optional Punc file for Punctuation dawg. Default: $(PUNC_FILE)
PUNC_FILE := $(OUTPUT_DIR)/$(MODEL_NAME).punc

# Name of the model to continue from. Default: '$(START_MODEL)'
START_MODEL =

LAST_CHECKPOINT = $(OUTPUT_DIR)/checkpoints/$(MODEL_NAME)_checkpoint

# Name of the proto model. Default: '$(PROTO_MODEL)'
PROTO_MODEL = $(OUTPUT_DIR)/$(MODEL_NAME).traineddata

# Tesseract model repo to use. Default: $(TESSDATA_REPO)
TESSDATA_REPO = _best

# If EPOCHS is given, it is used to set MAX_ITERATIONS.
ifeq ($(EPOCHS),)
# Max iterations. Default: $(MAX_ITERATIONS)
MAX_ITERATIONS := 10000
else
MAX_ITERATIONS := -$(EPOCHS)
endif

# Debug Interval. Default:  $(DEBUG_INTERVAL)
DEBUG_INTERVAL := 0

# Learning rate. Default: $(LEARNING_RATE)
ifdef START_MODEL
LEARNING_RATE := 0.0001
else
LEARNING_RATE := 0.002
endif

# Network specification. Default: $(NET_SPEC)
NET_SPEC := [1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx192 O1c\#\#\#]

# GPU/CUDA Support Settings
# Enable GPU acceleration if Tesseract is built with CUDA support. Default: $(USE_GPU)
USE_GPU ?= 0
# GPU device ID to use (for multi-GPU systems). Default: $(GPU_DEVICE)
GPU_DEVICE ?= 0
# Number of OpenMP threads for CPU parallelization. Default: auto-detected or 4
OMP_NUM_THREADS ?= $(shell nproc 2>/dev/null || echo 4)

# GPU Optimization Parameters
# Maximum GPU memory to use (in MB). Higher values = more parallel processing (faster).
# For T4 GPU (16GB): 14000 recommended. For larger GPUs: up to 28000+. Default: $(GPU_MAX_MEMORY)
# This controls batch processing - more memory = more images processed in parallel.
GPU_MAX_MEMORY ?= 14000
# Network mode for LSTM: 0=serial (slow, less memory), 1=parallel (fast, more memory). Default: $(NET_MODE)
NET_MODE ?= 1
# Append index for multi-GPU training (-1 for auto, 0+ for specific index). Default: $(APPEND_INDEX)
APPEND_INDEX ?= -1

# Advanced Training Parameters
# Perfect sample delay - iterations before using perfect samples (0=disabled, >0=delay). Default: $(PERFECT_SAMPLE_DELAY)
PERFECT_SAMPLE_DELAY ?= 0
# Weight range for initialization (higher=more random). Default: $(WEIGHT_RANGE)
WEIGHT_RANGE ?= 0.1
# Momentum for gradient descent (0.0-1.0, higher=more momentum). Default: $(MOMENTUM)
MOMENTUM ?= 0.9

TESSERACT_SCRIPTS := Arabic Armenian Bengali Bopomofo Canadian_Aboriginal Cherokee Cyrillic
TESSERACT_SCRIPTS += Devanagari Ethiopic Georgian Greek Gujarati Gurmukhi
TESSERACT_SCRIPTS += Hangul Han Hebrew Hiragana Kannada Katakana Khmer Lao Latin
TESSERACT_SCRIPTS += Malayalam Myanmar Ogham Oriya Runic Sinhala Syriac Tamil Telugu Thai

TESSERACT_LANGDATA = $(LANGDATA_DIR)/radical-stroke.txt $(TESSERACT_SCRIPTS:%=$(LANGDATA_DIR)/%.unicharset)

# Language Type - Indic, RTL or blank. Default: '$(LANG_TYPE)'
LANG_TYPE ?=

# Normalization mode - 2, 1 - for unicharset_extractor and Pass through Recoder for combine_lang_model
ifeq ($(LANG_TYPE),Indic)
	NORM_MODE =2
	RECODER =--pass_through_recoder
	GENERATE_BOX_SCRIPT =generate_wordstr_box.py
else
ifeq ($(LANG_TYPE),RTL)
	NORM_MODE =3
	RECODER =--pass_through_recoder --lang_is_rtl
	GENERATE_BOX_SCRIPT =generate_wordstr_box.py
else
	NORM_MODE =2
	RECODER=
	GENERATE_BOX_SCRIPT =generate_line_box.py
endif
endif

# Page segmentation mode. Default: $(PSM)
PSM = 13

# Random seed for shuffling of the training data. Default: $(RANDOM_SEED)
RANDOM_SEED := 0

# Ratio of train / eval training data. Default: $(RATIO_TRAIN)
RATIO_TRAIN := 0.90

# Default Target Error Rate. Default: $(TARGET_ERROR_RATE)
TARGET_ERROR_RATE := 0.01

# Use current Python program name on Windows
ifeq ($(OS),Windows_NT)
    PY_CMD := python
else
    PY_CMD := python3
endif

LOG_FILE = $(OUTPUT_DIR)/training.log

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    unicharset       Create unicharset"
	@echo "    charfreq         Show character histogram"
	@echo "    lists            Create lists of lstmf filenames for training and eval"
	@echo "    training         Start training (i.e. create .checkpoint files)"
	@echo "    traineddata      Create best and fast .traineddata files from each .checkpoint file"
	@echo "    proto-model      Build the proto model"
	@echo "    tesseract-langdata  Download stock unicharsets"
	@echo "    evaluation       Evaluate .checkpoint models on eval dataset via lstmeval"
	@echo "    plot             Generate train/eval error rate charts from training log"
	@echo "    clean-box        Clean generated .box files"
	@echo "    clean-lstmf      Clean generated .lstmf files"
	@echo "    clean-output     Clean generated output files"
	@echo "    clean            Clean all generated files"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    TESSDATA           Path to the directory containing START_MODEL.traineddata"
	@echo "                       (for example tesseract-ocr/tessdata_best). Default: $(TESSDATA)"
	@echo "    MODEL_NAME         Name of the model to be built. Default: $(MODEL_NAME)"
	@echo "    DATA_DIR           Data directory for output files, proto model, start model, etc. Default: $(DATA_DIR)"
	@echo "    LANGDATA_DIR       Data directory for langdata (downloaded from Tesseract langdata repo). Default: $(LANGDATA_DIR)"
	@echo "    OUTPUT_DIR         Output directory for generated files. Default: $(OUTPUT_DIR)"
	@echo "    GROUND_TRUTH_DIR   Ground truth directory. Default: $(GROUND_TRUTH_DIR)"
	@echo "    WORDLIST_FILE      Optional Wordlist file for Dictionary dawg. Default: $(WORDLIST_FILE)"
	@echo "    NUMBERS_FILE       Optional Numbers file for number patterns dawg. Default: $(NUMBERS_FILE)"
	@echo "    PUNC_FILE          Optional Punc file for Punctuation dawg. Default: $(PUNC_FILE)"
	@echo "    START_MODEL        Name of the model to continue from (i.e. fine-tune). Default: $(START_MODEL)"
	@echo "    PROTO_MODEL        Name of the prototype model. Default: $(PROTO_MODEL)"
	@echo "    TESSDATA_REPO      Tesseract model repo to use (_fast or _best). Default: $(TESSDATA_REPO)"
	@echo "    MAX_ITERATIONS     Max iterations. Default: $(MAX_ITERATIONS)"
	@echo "    EPOCHS             Set max iterations based on the number of lines for the training. Default: none"
	@echo "    DEBUG_INTERVAL     Debug Interval. Default:  $(DEBUG_INTERVAL)"
	@echo "    LEARNING_RATE      Learning rate. Default: $(LEARNING_RATE)"
	@echo "    NET_SPEC           Network specification (in VGSL) for new model from scratch. Default: $(NET_SPEC)"
	@echo "    LANG_TYPE          Language Type - Indic, RTL or blank. Default: '$(LANG_TYPE)'"
	@echo "    PSM                Page segmentation mode. Default: $(PSM)"
	@echo "    RANDOM_SEED        Random seed for shuffling of the training data. Default: $(RANDOM_SEED)"
	@echo "    RATIO_TRAIN        Ratio of train / eval training data. Default: $(RATIO_TRAIN)"
	@echo "    TARGET_ERROR_RATE  Default Target Error Rate. Default: $(TARGET_ERROR_RATE)"
	@echo "    LOG_FILE           File to copy training output to and read plot figures from. Default: $(LOG_FILE)"
	@echo ""
	@echo "  Performance Optimization Variables"
	@echo ""
	@echo "    USE_GPU            Enable GPU acceleration (requires CUDA-enabled Tesseract). Default: $(USE_GPU)"
	@echo "    GPU_DEVICE         GPU device ID to use for training (multi-GPU systems). Default: $(GPU_DEVICE)"
	@echo "    OMP_NUM_THREADS    Number of OpenMP threads for CPU parallelization. Default: $(OMP_NUM_THREADS)"
	@echo "    GPU_MAX_MEMORY     Maximum GPU memory in MB. Higher = faster (more parallel processing)."
	@echo "                       For T4 (16GB): 14000 recommended. Default: $(GPU_MAX_MEMORY)"
	@echo "    NET_MODE           LSTM network mode: 0=serial (slower, less memory), 1=parallel (faster). Default: $(NET_MODE)"
	@echo "    APPEND_INDEX       Multi-GPU training index (-1=auto, 0+=specific). Default: $(APPEND_INDEX)"
	@echo ""
	@echo "  Advanced Training Parameters"
	@echo ""
	@echo "    PERFECT_SAMPLE_DELAY  Iterations before using perfect samples (0=disabled). Default: $(PERFECT_SAMPLE_DELAY)"
	@echo "    WEIGHT_RANGE       Weight initialization range (higher=more random). Default: $(WEIGHT_RANGE)"
	@echo "    MOMENTUM           Gradient descent momentum (0.0-1.0). Default: $(MOMENTUM)"
	@echo ""
	@echo "  Note: lstmtraining doesn't have explicit batch size parameter. Instead, it processes"
	@echo "        images in parallel based on GPU_MAX_MEMORY. Higher memory = more images processed"
	@echo "        simultaneously = faster training. Monitor GPU usage with: watch nvidia-smi"

# END-EVAL

ifeq (4.2, $(firstword $(sort $(MAKE_VERSION) 4.2)))
# stuff that requires make-3.81 or higher
$(info You are using make version: $(MAKE_VERSION))
else
$(error This version of GNU Make is too low ($(MAKE_VERSION)). Check your path, or upgrade to 4.2 or newer.)
endif

.PRECIOUS: $(LAST_CHECKPOINT)

.PHONY: clean help lists proto-model tesseract-langdata training unicharset charfreq

ALL_FILES = $(and $(wildcard $(GROUND_TRUTH_DIR)),$(shell find -L $(GROUND_TRUTH_DIR) -name '*.gt.txt'))
unexport ALL_FILES # prevent adding this to envp in recipes (which can cause E2BIG if too long; cf. make #44853)
ALL_GT = $(OUTPUT_DIR)/all-gt
ALL_LSTMF = $(OUTPUT_DIR)/all-lstmf

# Create unicharset
unicharset: $(OUTPUT_DIR)/unicharset

# Show character histogram
charfreq: $(ALL_GT)
	LC_ALL=C.UTF-8 grep -P -o "\X" $< | sort | uniq -c | sort -rn

# Create lists of lstmf filenames for training and eval
lists: $(OUTPUT_DIR)/list.train $(OUTPUT_DIR)/list.eval

$(OUTPUT_DIR):
	@mkdir -p $@

$(OUTPUT_DIR)/list.eval \
$(OUTPUT_DIR)/list.train: $(ALL_LSTMF) | $(OUTPUT_DIR)
	$(PY_CMD) generate_eval_train.py $(ALL_LSTMF) $(RATIO_TRAIN)

ifdef START_MODEL
$(DATA_DIR)/$(START_MODEL)/$(MODEL_NAME).lstm-unicharset:
	@mkdir -p $(@D)
	combine_tessdata -u $(TESSDATA)/$(START_MODEL).traineddata $(basename $@)
$(OUTPUT_DIR)/my.unicharset: $(ALL_GT) | $(OUTPUT_DIR)
	unicharset_extractor --output_unicharset "$@" --norm_mode $(NORM_MODE) "$^"
$(OUTPUT_DIR)/unicharset: $(DATA_DIR)/$(START_MODEL)/$(MODEL_NAME).lstm-unicharset $(OUTPUT_DIR)/my.unicharset
	merge_unicharsets $^ "$@"
else
$(OUTPUT_DIR)/unicharset: $(ALL_GT) | $(OUTPUT_DIR)
	unicharset_extractor --output_unicharset "$@" --norm_mode $(NORM_MODE) "$(ALL_GT)"
endif

# Start training
training: $(OUTPUT_DIR).traineddata

$(ALL_GT): $(ALL_FILES) | $(OUTPUT_DIR)
	$(if $^,,$(error found no $(GROUND_TRUTH_DIR)/*.gt.txt for $@))
	$(file >$@) $(foreach F,$^,$(file >>$@,$(file <$F)))

.PRECIOUS: %.box
%.box: %.png %.gt.txt
	PYTHONIOENCODING=utf-8 $(PY_CMD) $(GENERATE_BOX_SCRIPT) -i "$*.png" -t "$*.gt.txt" > "$@"

%.box: %.bin.png %.gt.txt
	PYTHONIOENCODING=utf-8 $(PY_CMD) $(GENERATE_BOX_SCRIPT) -i "$*.bin.png" -t "$*.gt.txt" > "$@"

%.box: %.nrm.png %.gt.txt
	PYTHONIOENCODING=utf-8 $(PY_CMD) $(GENERATE_BOX_SCRIPT) -i "$*.nrm.png" -t "$*.gt.txt" > "$@"

%.box: %.raw.png %.gt.txt
	PYTHONIOENCODING=utf-8 $(PY_CMD) $(GENERATE_BOX_SCRIPT) -i "$*.raw.png" -t "$*.gt.txt" > "$@"

%.box: %.tif %.gt.txt
	PYTHONIOENCODING=utf-8 $(PY_CMD) $(GENERATE_BOX_SCRIPT) -i "$*.tif" -t "$*.gt.txt" > "$@"

$(ALL_LSTMF): $(ALL_FILES:%.gt.txt=%.lstmf)
	$(if $^,,$(error found no $(GROUND_TRUTH_DIR)/*.lstmf for $@))
	@mkdir -p $(@D)
	$(file >$@) $(foreach F,$^,$(file >>$@,$F))
	$(PY_CMD) shuffle.py $(RANDOM_SEED) "$@"

.PRECIOUS: %.lstmf
%.lstmf: %.png %.box
	tesseract "$<" $* --psm $(PSM) lstm.train

%.lstmf: %.bin.png %.box
	tesseract "$<" $* --psm $(PSM) lstm.train

%.lstmf: %.nrm.png %.box
	tesseract "$<" $* --psm $(PSM) lstm.train

%.lstmf: %.raw.png %.box
	tesseract "$<" $* --psm $(PSM) lstm.train

%.lstmf: %.tif %.box
	tesseract "$<" $* --psm $(PSM) lstm.train

.PHONY: traineddata
CHECKPOINT_FILES = $(wildcard $(OUTPUT_DIR)/checkpoints/$(MODEL_NAME)*.checkpoint)
BESTMODEL_FILES = $(subst checkpoints,tessdata_best,$(CHECKPOINT_FILES:%.checkpoint=%.traineddata))
FASTMODEL_FILES = $(subst checkpoints,tessdata_fast,$(CHECKPOINT_FILES:%.checkpoint=%.traineddata))
# Create best and fast .traineddata files from each .checkpoint file
traineddata: $(BESTMODEL_FILES)
traineddata: $(FASTMODEL_FILES)
$(OUTPUT_DIR)/tessdata_best $(OUTPUT_DIR)/tessdata_fast $(OUTPUT_DIR)/eval:
	@mkdir -p $@
$(OUTPUT_DIR)/tessdata_best/%.traineddata: $(OUTPUT_DIR)/checkpoints/%.checkpoint | $(OUTPUT_DIR)/tessdata_best
	$(if $(filter 1,$(USE_GPU)),OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) TESSERACT_OPENCL_DEVICE=GPU:$(GPU_DEVICE),OMP_NUM_THREADS=$(OMP_NUM_THREADS)) \
	lstmtraining \
          --stop_training \
          --continue_from $< \
          --traineddata $(PROTO_MODEL) \
          --model_output $@
$(OUTPUT_DIR)/tessdata_fast/%.traineddata: $(OUTPUT_DIR)/checkpoints/%.checkpoint | $(OUTPUT_DIR)/tessdata_fast
	$(if $(filter 1,$(USE_GPU)),OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) TESSERACT_OPENCL_DEVICE=GPU:$(GPU_DEVICE),OMP_NUM_THREADS=$(OMP_NUM_THREADS)) \
	lstmtraining \
          --stop_training \
          --continue_from $< \
          --traineddata $(PROTO_MODEL) \
          --convert_to_int \
          --model_output $@

# Build the proto model
proto-model: $(PROTO_MODEL)

$(PROTO_MODEL): $(OUTPUT_DIR)/unicharset $(TESSERACT_LANGDATA)
ifeq (Windows_NT, $(OS))
	- dos2unix "$(NUMBERS_FILE)"
	- dos2unix "$(PUNC_FILE)"
	- dos2unix "$(WORDLIST_FILE)"
	- dos2unix "$(LANGDATA_DIR)/$(MODEL_NAME)/$(MODEL_NAME).config"
endif
	$(if $(filter-out $(abspath $@),$(abspath $(DATA_DIR)/$(MODEL_NAME)/$(MODEL_NAME).traineddata)),\
	$(error $@!=$(DATA_DIR)/$(MODEL_NAME)/$(MODEL_NAME).traineddata -- consider setting different values for DATA_DIR, OUTPUT_DIR, or PROTO_MODEL))
	combine_lang_model \
	  --input_unicharset $(OUTPUT_DIR)/unicharset \
	  --script_dir $(LANGDATA_DIR) \
	  --numbers $(NUMBERS_FILE) \
	  --puncs $(PUNC_FILE) \
	  --words $(WORDLIST_FILE) \
	  --output_dir $(DATA_DIR) \
	  $(RECODER) \
	  --lang $(MODEL_NAME)

ifdef START_MODEL
$(LAST_CHECKPOINT): unicharset lists $(PROTO_MODEL)
	@mkdir -p $(OUTPUT_DIR)/checkpoints
	@echo
	@echo "=== Training Configuration ==="
	@echo "GPU Acceleration: $(if $(filter 1,$(USE_GPU)),ENABLED (Device $(GPU_DEVICE)) - FORCING GPU ONLY,DISABLED)"
	@echo "$(if $(filter 1,$(USE_GPU)),GPU Max Memory: $(GPU_MAX_MEMORY) MB,)"
	@echo "$(if $(filter 1,$(USE_GPU)),Network Mode: $(if $(filter 1,$(NET_MODE)),Parallel (Fast),Serial (Memory-efficient)),)"
	@echo "OpenMP Threads: $(if $(filter 1,$(USE_GPU)),1 (GPU mode),$(OMP_NUM_THREADS))"
	@echo "Learning Rate: $(LEARNING_RATE)"
	@echo "Max Iterations: $(MAX_ITERATIONS)"
	@echo "=============================="
	@echo
	@if [ "$(USE_GPU)" = "1" ]; then \
		echo "⚠️  GPU Mode Enabled - Verifying GPU availability..."; \
		if ! command -v nvidia-smi >/dev/null 2>&1; then \
			echo "❌ WARNING: nvidia-smi not found. GPU drivers may not be installed."; \
			echo "   Training may fall back to CPU despite USE_GPU=1 setting."; \
			echo "   Install NVIDIA drivers to enable GPU acceleration."; \
		elif ! nvidia-smi >/dev/null 2>&1; then \
			echo "❌ WARNING: nvidia-smi failed. No GPUs detected or drivers not working."; \
			echo "   Training will likely fall back to CPU."; \
		else \
			echo "✓ GPU detected:"; \
			nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (GPU info unavailable)"; \
		fi; \
		if ! command -v clinfo >/dev/null 2>&1; then \
			echo "⚠️  clinfo not found. Cannot verify OpenCL support."; \
			echo "   Install clinfo: sudo apt-get install clinfo"; \
		fi; \
		echo ""; \
		echo "NOTE: For GPU to work, Tesseract must be built with --enable-opencl"; \
		echo "      If training uses CPU despite this message, rebuild Tesseract with OpenCL."; \
		echo ""; \
	fi
	$(if $(filter 1,$(USE_GPU)),OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) TESSERACT_OPENCL_DEVICE=GPU:$(GPU_DEVICE),OMP_NUM_THREADS=$(OMP_NUM_THREADS)) \
	lstmtraining \
	  --debug_interval $(DEBUG_INTERVAL) \
	  --traineddata $(PROTO_MODEL) \
	  --old_traineddata $(TESSDATA)/$(START_MODEL).traineddata \
	  --continue_from $(DATA_DIR)/$(START_MODEL)/$(MODEL_NAME).lstm \
	  --learning_rate $(LEARNING_RATE) \
	  --model_output $(OUTPUT_DIR)/checkpoints/$(MODEL_NAME) \
	  --train_listfile $(OUTPUT_DIR)/list.train \
	  --eval_listfile $(OUTPUT_DIR)/list.eval \
	  --max_iterations $(MAX_ITERATIONS) \
	  --target_error_rate $(TARGET_ERROR_RATE) \
	  $(if $(filter 1,$(USE_GPU)),--max_image_MB $(GPU_MAX_MEMORY) --net_mode $(NET_MODE),) \
	  $(if $(and $(filter 1,$(USE_GPU)),$(filter-out -1,$(APPEND_INDEX))),--append_index $(APPEND_INDEX),)$(if $(filter-out 0,$(PERFECT_SAMPLE_DELAY)), --perfect_sample_delay $(PERFECT_SAMPLE_DELAY),)$(if $(WEIGHT_RANGE), --weight_range $(WEIGHT_RANGE),)$(if $(MOMENTUM), --momentum $(MOMENTUM),) \
	2>&1 | tee -a $(LOG_FILE)
$(OUTPUT_DIR).traineddata: $(LAST_CHECKPOINT)
	@echo
	$(if $(filter 1,$(USE_GPU)),OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) TESSERACT_OPENCL_DEVICE=GPU:$(GPU_DEVICE),OMP_NUM_THREADS=$(OMP_NUM_THREADS)) \
	lstmtraining \
	--stop_training \
	--continue_from $(LAST_CHECKPOINT) \
	--traineddata $(PROTO_MODEL) \
	--model_output $@
else
$(LAST_CHECKPOINT): unicharset lists $(PROTO_MODEL)
	@mkdir -p $(OUTPUT_DIR)/checkpoints
	@echo
	@echo "=== Training Configuration ==="
	@echo "GPU Acceleration: $(if $(filter 1,$(USE_GPU)),ENABLED (Device $(GPU_DEVICE)) - FORCING GPU ONLY,DISABLED)"
	@echo "$(if $(filter 1,$(USE_GPU)),GPU Max Memory: $(GPU_MAX_MEMORY) MB,)"
	@echo "$(if $(filter 1,$(USE_GPU)),Network Mode: $(if $(filter 1,$(NET_MODE)),Parallel (Fast),Serial (Memory-efficient)),)"
	@echo "OpenMP Threads: $(if $(filter 1,$(USE_GPU)),1 (GPU mode),$(OMP_NUM_THREADS))"
	@echo "Learning Rate: $(LEARNING_RATE)"
	@echo "Max Iterations: $(MAX_ITERATIONS)"
	@echo "=============================="
	@echo
	@if [ "$(USE_GPU)" = "1" ]; then \
		echo "⚠️  GPU Mode Enabled - Verifying GPU availability..."; \
		if ! command -v nvidia-smi >/dev/null 2>&1; then \
			echo "❌ WARNING: nvidia-smi not found. GPU drivers may not be installed."; \
			echo "   Training may fall back to CPU despite USE_GPU=1 setting."; \
			echo "   Install NVIDIA drivers to enable GPU acceleration."; \
		elif ! nvidia-smi >/dev/null 2>&1; then \
			echo "❌ WARNING: nvidia-smi failed. No GPUs detected or drivers not working."; \
			echo "   Training will likely fall back to CPU."; \
		else \
			echo "✓ GPU detected:"; \
			nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (GPU info unavailable)"; \
		fi; \
		if ! command -v clinfo >/dev/null 2>&1; then \
			echo "⚠️  clinfo not found. Cannot verify OpenCL support."; \
			echo "   Install clinfo: sudo apt-get install clinfo"; \
		fi; \
		echo ""; \
		echo "NOTE: For GPU to work, Tesseract must be built with --enable-opencl"; \
		echo "      If training uses CPU despite this message, rebuild Tesseract with OpenCL."; \
		echo ""; \
	fi
	$(if $(filter 1,$(USE_GPU)),OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) TESSERACT_OPENCL_DEVICE=GPU:$(GPU_DEVICE),OMP_NUM_THREADS=$(OMP_NUM_THREADS)) \
	lstmtraining \
	  --debug_interval $(DEBUG_INTERVAL) \
	  --traineddata $(PROTO_MODEL) \
	  --learning_rate $(LEARNING_RATE) \
	  --net_spec "$(subst c###,c$(firstword $(file <$(OUTPUT_DIR)/unicharset)),$(NET_SPEC))" \
	  --model_output $(OUTPUT_DIR)/checkpoints/$(MODEL_NAME) \
	  --train_listfile $(OUTPUT_DIR)/list.train \
	  --eval_listfile $(OUTPUT_DIR)/list.eval \
	  --max_iterations $(MAX_ITERATIONS) \
	  --target_error_rate $(TARGET_ERROR_RATE) \
	  $(if $(filter 1,$(USE_GPU)),--max_image_MB $(GPU_MAX_MEMORY) --net_mode $(NET_MODE),) \
	  $(if $(and $(filter 1,$(USE_GPU)),$(filter-out -1,$(APPEND_INDEX))),--append_index $(APPEND_INDEX),)$(if $(filter-out 0,$(PERFECT_SAMPLE_DELAY)), --perfect_sample_delay $(PERFECT_SAMPLE_DELAY),)$(if $(WEIGHT_RANGE), --weight_range $(WEIGHT_RANGE),)$(if $(MOMENTUM), --momentum $(MOMENTUM),) \
	2>&1 | tee -a $(LOG_FILE)
$(OUTPUT_DIR).traineddata: $(LAST_CHECKPOINT)
	@echo
	$(if $(filter 1,$(USE_GPU)),OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) TESSERACT_OPENCL_DEVICE=GPU:$(GPU_DEVICE),OMP_NUM_THREADS=$(OMP_NUM_THREADS)) \
	lstmtraining \
	--stop_training \
	--continue_from $(LAST_CHECKPOINT) \
	--traineddata $(PROTO_MODEL) \
	--model_output $@
endif

# plotting

# Build lstmeval files list based on respective best traineddata models
BEST_LSTMEVAL_FILES = $(subst tessdata_best,eval,$(BESTMODEL_FILES:%.traineddata=%.eval.log))
$(BEST_LSTMEVAL_FILES): $(OUTPUT_DIR)/eval/%.eval.log: $(OUTPUT_DIR)/tessdata_best/%.traineddata | $(OUTPUT_DIR)/eval
ifeq ($(OS),Windows_NT)
    # Windows with PowerShell timing
	powershell -Command "Measure-Command { \
		lstmeval  \
			--verbosity=0 \
			--model $< \
			--eval_listfile $(OUTPUT_DIR)/list.eval 2>&1 | Select-String '^BCER eval' | Out-File $@ \
	}"
else
    # Unix-like systems
	time -p lstmeval  \
		--verbosity=0 \
		--model $< \
		--eval_listfile $(OUTPUT_DIR)/list.eval 2>&1 | grep "^BCER eval" > $@
endif
# Make TSV with lstmeval CER and checkpoint filename parts
TSV_LSTMEVAL = $(OUTPUT_DIR)/lstmeval.tsv
.INTERMEDIATE: $(TSV_LSTMEVAL)
$(TSV_LSTMEVAL): $(BEST_LSTMEVAL_FILES)
	@echo "Name	CheckpointCER	LearningIteration	TrainingIteration	EvalCER	IterationCER	SubtrainerCER" > "$@"
	@{ $(foreach F,$^,echo -n "$F "; grep BCER $F;) } | sort -rn | \
	sed -e 's|^$(OUTPUT_DIR)/eval/$(MODEL_NAME)_\([0-9.]*\)_\([0-9]*\)_\([0-9]*\).eval.log BCER eval=\([0-9.]*\).*$$|\t\1\t\2\t\3\t\4\t\t|' >>  "$@"
# Make TSV with CER at every 100 iterations.
TSV_100_ITERATIONS = $(OUTPUT_DIR)/iteration.tsv
.INTERMEDIATE: $(TSV_100_ITERATIONS)
$(TSV_100_ITERATIONS): $(LOG_FILE)
	@echo "Name	CheckpointCER	LearningIteration	TrainingIteration	EvalCER	IterationCER	SubtrainerCER" > "$@"
	@grep 'At iteration' $< \
		| sed -e '/^Sub/d' \
		| sed -e '/^Update/d' \
		| sed -e '/^ New worst BCER/d' \
		| sed -e 's|At iteration \([0-9]*\)/\([0-9]*\)/.*BCER train=|\t\t\1\t\2\t\t|' \
		| sed -e 's/%, BWER.*/\t/' >>  "$@"
# Make TSV with Checkpoint CER.
TSV_CHECKPOINT = $(OUTPUT_DIR)/checkpoint.tsv
.INTERMEDIATE: $(TSV_CHECKPOINT)
$(TSV_CHECKPOINT): $(LOG_FILE)
	@echo "Name	CheckpointCER	LearningIteration	TrainingIteration	EvalCER	IterationCER	SubtrainerCER" > "$@"
	@grep 'best model' $< \
		| sed -e 's/^.*\///' \
		| sed -e 's/\.checkpoint.*$$/\t\t\t/' \
		| sed -e 's/_/\t/g' >>  "$@"
# Make TSV with Eval CER.
TSV_EVAL = $(OUTPUT_DIR)/eval.tsv
.INTERMEDIATE: $(TSV_EVAL)
$(TSV_EVAL): $(LOG_FILE)
	@echo "Name	CheckpointCER	LearningIteration	TrainingIteration	EvalCER	IterationCER	SubtrainerCER" > "$@"
	@grep 'BCER eval' $< \
		| sed -e 's/^.*[0-9]At iteration //' \
		| sed -e 's/,.* BCER eval=/\t\t/'  \
		| sed -e 's/, BWER.*$$/\t\t/' \
		| sed -e 's/^/\t\t/' >>  "$@"
# Make TSV with Subtrainer CER.
TSV_SUB = $(OUTPUT_DIR)/sub.tsv
.INTERMEDIATE: $(TSV_SUB)
$(TSV_SUB): $(LOG_FILE)
	@echo "Name	CheckpointCER	LearningIteration	TrainingIteration	EvalCER	IterationCER	SubtrainerCER" > "$@"
	@grep '^UpdateSubtrainer' $< \
		| sed -e 's/^.*At iteration \([0-9]*\)\/\([0-9]*\)\/.*BCER train=/\t\t\1\t\2\t\t\t/' \
		| sed -e 's/%, BWER.*//' >>  "$@"

$(OUTPUT_DIR)/$(MODEL_NAME).plot_log.png: $(TSV_100_ITERATIONS) $(TSV_CHECKPOINT) $(TSV_EVAL) $(TSV_SUB)
	$(PY_CMD) plot_log.py $@ $(MODEL_NAME) $^
$(OUTPUT_DIR)/$(MODEL_NAME).plot_cer.png: $(TSV_100_ITERATIONS) $(TSV_CHECKPOINT) $(TSV_EVAL) $(TSV_SUB) $(TSV_LSTMEVAL)
	$(PY_CMD) plot_cer.py $@ $(MODEL_NAME) $^

.PHONY: evaluation plot
# run lstmeval on list.eval data for each checkpoint model
evaluation: $(BEST_LSTMEVAL_FILES)
# combine TSV files with all required CER values, generated from training log and validation logs, then plot
plot: $(OUTPUT_DIR)/$(MODEL_NAME).plot_cer.png $(OUTPUT_DIR)/$(MODEL_NAME).plot_log.png


tesseract-langdata: $(TESSERACT_LANGDATA)

$(TESSERACT_LANGDATA):
	@mkdir -p $(@D)
	wget -O $@ 'https://github.com/tesseract-ocr/langdata_lstm/raw/main/$(@F)'

$(TESSDATA)/%.traineddata:
	wget -O $@ 'https://github.com/tesseract-ocr/tessdata$(TESSDATA_REPO)/raw/main/$(@F)'

# Clean generated .box files
.PHONY: clean-box
clean-box:
	find -L $(GROUND_TRUTH_DIR) -name '*.box' -delete

# Clean generated .lstmf files
.PHONY: clean-lstmf
clean-lstmf:
	find -L $(GROUND_TRUTH_DIR) -name '*.lstmf' -delete

# Clean generated output files
.PHONY: clean-output
clean-output:
	rm -rf $(OUTPUT_DIR)

# Clean all generated files
clean: clean-box clean-lstmf clean-output
