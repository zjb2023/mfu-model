#!/bin/bash
set -eo pipefail
set -x
#export TORCH_CPP_LOG_LEVEL=INFO

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
pip install -U /mnt/zj-data/models/public/yanjun/suzy-dev-0605/source/deep_ep-1.1.0+d7577a6-cp310-cp310-linux_x86_64.whl
# DEEPEP_TOKEN_NUM_MULTIPLIER
# Runs the "175B" parameter model
# source /mnt/moer-train/public/1000B/TransformerEngine/install_wheel.sh
# Please change the following envrioment variables
# base on the cluster configuration

# WORK_HOME=/mnt/moer-train/public/fukejie/1k+/MT-MegatronLM/examples/kimi-k2
# PATCH_HOME=/mnt/moer-train/public/fukejie/1k+/MT-MegatronLM
# WORK_HOME=/mnt/moer-train/public/suzy/dev-0507/MT-MegatronLM/examples/kimi-k2
# PATCH_HOME=/mnt/moer-train/public/suzy/dev-0507/MT-MegatronLM
WORK_HOME=/mnt/zj-data/data/yu/236B/megatron-lm-musa-patch/examples/kimi-k2/kimi-k2
PATCH_HOME=/mnt/zj-data/data/yu/236B/megatron-lm-musa-patch

TP_SIZE=${TP:-1}
PP_SIZE=${PP:-16}
EP_SIZE=${EP:-8}
CP_SIZE=${CP:-2}
FP8=${FP8:-false}
PP_LAYOUT=none
############################## test options ###################
TOKEN_DISPATCHER=${TOKEN_DISPATCHER:-deepep}
USE_DEEPEP_ACE=1
# CE_BUFFER_MULTIPLE=2
AC=${AC:-full}
SWIGLU_FUSION=${SWIGLU_FUSION:-false}
PAO=${PAO:-moments}
SE_OVERLAP=${SE_OVERLAP:-false}
USE_TOPK_ROUTER_FUSION=${USE_TOPK_ROUTER_FUSION:-false}
# USE_OPTIMIZED_FUSED_MULTIHOT=${USE_OPTIMIZED_FUSED_MULTIHOT:-0}

############################## end of test options ############
MTP=${MTP:-1}
FORCE_LB=${FORCE_LB:-false}
EXIT_INTERVAL=${EXIT_INTERVAL:-100}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}

#NUM_NODES=${WORLD_SIZE:-2}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

TOKENIZED_MODEL=/mnt/zj-data/data/yu/zjllm-llama3-tokenizer
CURRENT_TIME=$(date "+%Y-%m-%d_%H%M")
RDZV_ID=$CURRENT_TIME"_full_moments"
EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}_mtp${MTP}_forcelb${FORCE_LB}_pertensor${FP8}_NO_LOSS_REDUCE"

#export ENABLE_PROFILER=0
#export PROFILER_FREQ=10
#export PROFILER_SAVE_DIR=${PROFILER_SAVE_DIR:-"/mnt/zj-data/data/yu/output/trace/236-newframe"}
#export PROFILER_WITH_STACK=1
#------------------------------------------comm_profiling--------------------------------------------
#export MCCL_DEBUG=INFO
#export MCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,COLL,NET
#export MCCL_DEBUG_TIMESTAMP_FORMAT="[%F %T.%6f] "
#export MCCL_DEBUG_TIMESTAMP_LEVELS=ALL
X10K_DATA_ROOT=/mnt/zj-data/data/yu/x10000_data_output/$(date '+%Y-%m-%d-%H:%M')/$(hostname)
OUTPUT_DIR=$X10K_DATA_ROOT
#profiler
export ENABLE_PROFILER=1
export PROFILER_FREQ=5
export PROFILER_SAVE_DIR=$X10K_DATA_ROOT/profiler
export PROFILER_WITH_STACK=0
#deepep
export DEBUG_DEEPEP=1
export MEASURE_DEEPEP_BW=1
export DEEPEP_LOG_FREQ=1
export DEEPEP_LOG_FILE=$X10K_DATA_ROOT/deepep_trace
#nic
#MAX_DATA_GIB=1 \
#RDMA_MIN_FREE_GIB=1 \
#nohup /mnt/si002962t2xy/default/work/rdma_counter_collector/bin/rdma_counter_ctl.sh start-long \
#  > /mnt/si002962t2xy/default/work/rdma_counter_collector/rdma_counter_nohup.log 2>&1 &
#-------------------------rdma
MAX_DATA_MIB=102400 \
RDMA_MIN_FREE_MIB=1024 \
nohup /mnt/zj-data/data/xutk/work/rdma_counter_collector/bin/rdma_counter_ctl.sh start-long \
  > /mnt/zj-data/data/xutk/work/rdma_counter_collector/rdma_counter_nohup.log 2>&1 &
#-----------------------mtlink
MAX_DATA_MIB=102400 \
MTLINK_MIN_FREE_MIB=1024 \
nohup /mnt/zj-data/data/xutk/work/mtlink_counter_collector/bin/mtlink_counter_ctl.sh start-long \
  -P 0 \
  > /mnt/zj-data/data/xutk/work/mtlink_counter_collector/mtlink_counter_nohup.log 2>&1 &

# export MUSA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_EXECUTION_TIMEOUT=480000
# export MUSA_EXECUTION_TIMEOUT=86400000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MCCL_IB_GID_INDEX=3
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_ALGOS=1
# export MCCL_BUFFSIZE=20971520
export MCCL_BUFFSIZE=4194304
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_TC=136
export MCCL_IB_QPS_PER_CONNECTION=16
export MCCL_CROSS_NIC=0
export USE_RECOMPUTE_VARIANCE=0
export ENABLE_D2H_IN_PERMUTATION=0
export NO_LOSS_REDUCE=1
# export USE_MUSA_MOE=1
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
export MCCL_LIB=/usr/local/musa/lib/libmccl.so
export MUSA_ERROR_DUMP_PATH=/mnt/moer-train/users/suzy/output/error_dump/$(date "+%Y.%m.%d-%H:%M:%S")
MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH
export USE_DEEPEP_ACE=1
export EP_BALANCE_INFO=0
export MUSA_LOG=0x1
# export MCCL_MIN_NCHANNELS=4               # reduce finalize_model_grad time consumption. By Zhiyi 06.13.2026


# export DUMP_MEMORY_SNAPSHOT=1
# export MEMORY_SNAPSHOT_PATH=$WORK_HOME/mem_snapshot/$RDZV_ID
# mkdir -p $MEMORY_SNAPSHOT_PATH
# export RDZV_ID=$RDZV_ID


export TE_MULTI_STREAM_GROUPGEMM=1
if [ ! -d "${MEGATRON_PATH}/build" ]; then
    cd "${MEGATRON_PATH}"
    python setup.py build_ext --inplace
    cd -
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$OUTPUT_DIR/checkpoints"}
mkdir -p $CHECKPOINT_PATH
CHECKPOINT_LOAD_PATH=${CHECKPOINT_LOAD_PATH:-"$CHECKPOINT_PATH"}
# DATA_PATH=$DATA_DIR

#VMM
export PYTORCH_MUSA_ALLOC_CONF="expandable_segments:True"
export TORCH_MCCL_AVOID_RECORD_STREAMS=1

#export NODE_ADDR=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)
#export GPUS_PER_NODE=8
#export NODE_RANK=$RANK
#if [ -n "$OMPI_COMM_WORLD_RANK" ]; then
#    export NODE_RANK=$OMPI_COMM_WORLD_RANK
#else
#    export NODE_RANK=$RANK
#fi
# export MUSA_LAUNCH_BLOCKING=1
# export MCCL_DEBUG=INFO

#DATASET_FILE=${DATASET_FILE:-"/mnt/zj-data/data/suntianning/X10000/dataset/tokens/SlimPajama-627B/slimpajama627b_llama3_train.datalist"}
DATA_PATH=/mnt/zj-data/data/yu/tiny_qwen3_text_document/tiny_qwen3_text_document
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/zj-data/data/yu/data/datacache/"}
DATASET_FILE=${DATASET_FILE:-"/mnt/zj-data/data/suntianning/X10000/dataset/tokens/SlimPajama-627B/slimpajama627b_llama3_train.datalist"}
DATA_PATH="$(grep -v '^#' ${DATASET_FILE})"
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/zj-data/models/public/yanjun/data/datacache/"}
# DATASET_FILE=$DATASET_FILE
# DATA_PATH="$(grep -v '^#' ${DATASET_FILE})"
# DATA_CACHE_PATH=$DATA_CACHE_PATH
# TOTAL_TOKENS=20934786631107 #

cd "$(dirname "$0")"
TOTAL_TOKENS="$(($(python3 sum_row.py ${DATASET_FILE})))"

#TOTAL_TOKENS=209311600000

SEQ_LEN=${SEQ_LEN:-8192}    # 128k:131072 32k: 32768
SAMPLE_SIZE="$((${TOTAL_TOKENS}/${SEQ_LEN}))"
TRAIN_SAMPLES=$SAMPLE_SIZE
TRAIN_ITERS=$(( ${TOTAL_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
WARMUP_STEPS=0
WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))
# WARMUP_STEPS=2000
# WARMUP_SAMPLES=$((WARMUP_STEPS * 4500))

export NODE_ADDR=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)
export GPUS_PER_NODE=8

HOSTFILE=/etc/mpi/hostfile

# 读取host列表
mapfile -t HOSTS < <(awk '{print $1}' "${HOSTFILE}")

NUM_NODES=${#HOSTS[@]}

# 默认master就是hostfile第一台
export MASTER_ADDR="${HOSTS[0]}"

# MASTER_PORT保持外部传入，没有则使用29500
export MASTER_PORT=${MASTER_PORT:-16250}

THIS_HOST=$(hostname -f)

#MY_IP=$(hostname -i | awk '{print $1}')

#NODE_RANK=$(cat /etc/mpi/hostfile | while read host; do
#    getent hosts ${host} | awk '{print $1}'
#done | grep -n "^${MY_IP}" | cut -d: -f1)

#NODE_RANK=$((NODE_RANK-1))
MY_IP=$(hostname -i | awk '{print $1}')

#NODE_RANK=$(awk -v ip="${MY_IP}" '
#{
#    cmd="getent hosts "$1" | head -n1"
#    cmd | getline line
#    close(cmd)
#
#    split(line,a," ")
#
#    if (a[1] == ip) {
#        print NR-1
#        exit
#    }
#}' /etc/mpi/hostfile)
#MY_HOST=$(hostname)

#NODE_RANK=$(awk -v host="$MY_HOST" '
#{
#    if ($1 == host) {
#        print NR-1
#        exit
#    }
#}' /etc/mpi/hostfile)
MY_IP=$(hostname -i | awk '{print $1}')

NODE_RANK=$(awk -v ip="$MY_IP" '
{
    cmd="getent ahostsv4 "$1" | awk '\''NR==1 {print $1}'\''"
    cmd | getline host_ip
    close(cmd)

    if (host_ip == ip) {
        print NR-1
        exit
    }
}' /etc/mpi/hostfile)

echo "MY_IP=$MY_IP"
echo "NODE_RANK=$NODE_RANK"
echo "MY_IP=${MY_IP}"
echo "NODE_RANK=${NODE_RANK}"
RUN_DIR=${OUTPUT_DIR}/${CURRENT_TIME}/
mkdir -p "${RUN_DIR}"
LOG_FILE_TMP="${RUN_DIR}/${EXPNAME}.RANK${NODE_RANK}.${NODE_ADDR}.log"
LOG_FILE=${LOG_FILE:-${LOG_FILE_TMP}}

LOG_PATH=$OUTPUT_DIR/${CURRENT_TIME}/logs/$RDZV_ID
mkdir -p $LOG_PATH
# cp $0 $LOG_PATH/
TB_PATH=$OUTPUT_DIR/tf_logs/${CURRENT_TIME}/${MARK}/
mkdir -p $TB_PATH
WB_PATH=$OUTPUT_DIR/wandb/${CURRENT_TIME}/
mkdir -p $WB_PATH

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
#     --log_dir $LOG_FILE_TMP #$WORK_HOME/output_log/$RDZV_ID/$EXPNAME/$NODE_RANK
#     --redirects 3
)

LAYERS=${LAYERS:-60}
MODEL_ARGS=(
    --num-layers $LAYERS  # 61
    --hidden-size 5120
    --num-attention-heads 128
    --seq-length $SEQ_LEN
    --max-position-embeddings $SEQ_LEN
    --norm-epsilon 1e-6
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    --ffn-hidden-size 12288
    --position-embedding-type rope
    --swiglu
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --rope-type yarn
)

FIRST_STAGE=${FIRST_STAGE:-2}
LAST_STAGE=${LAST_STAGE:-2}

# 24414062 1T
TRAINING_ARGS=(
    --seed 1234
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    # --rampup-batch-size 3072 1536 114501954
    --train-samples $TRAIN_SAMPLES #2441406200
    --init-method-std  0.006
    --use-mcore-models
    # --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    # --no-rope-fusion
    # --no-bias-swiglu-fusion
    --use-distributed-optimizer
    --use-flash-attn
    --sequence-parallel
    # =================== recompute full schema =================
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers ${MP_AC_LAYERS:-1}
    # ===========================================================
    --distributed-backend nccl
    --multi-latent-attention
    --qk-layernorm
    --enable-experimental
    # --mlp-recompute
    # --mlp-rms-recompute
    # --recompute-variance
    # --attn-recompute
    # --mla-rms-recompute
    --manual-gc
    --manual-gc-interval 100
    --exit-interval ${EXIT_INTERVAL}
    --rotary-seq-len-interpolation-factor 1
)

if [[ $AC = full ]]; then
    TRAINING_ARGS+=(
        # =================== recompute full schema  128k增加=================
        --recompute-granularity full
        --recompute-method block
        --recompute-num-layers 4
        # ===========================================================
    )
elif [[ $AC = offload ]]; then
    TRAINING_ARGS+=(
        # =================== recompute selective schema ====================
        --recompute-granularity selective
        # --recompute-modules mla_up_proj moe_act layernorm mlp shared_experts
        --recompute-modules mla_up_proj moe_act layernorm mlp
        --offload-moe-fc1-input
        --offload-moe-fused-swiglu-input
        # ==================================================================
    )
elif [[ $AC = hybrid ]]; then
    TRAINING_ARGS+=(
        # =================== recompute hybrid schema ====================
        # stage 0: full AC (block method, skip last layer in pp_rank==0)
        # last stage: no recomputation at all
        # other stages: selective AC
        --recompute-granularity selective
        --recompute-num-layers 0       # signal for hybrid mode
        --recompute-modules mla_up_proj moe_act layernorm mlp shared_experts
        --offload-moe-fc1-input
        --offload-moe-fused-swiglu-input
        # ================================================================
    )
elif [[ $AC = 236test ]]; then
# =================== 236b pretrain ====================
    TRAINING_ARGS+=(
        --recompute-granularity selective
        --recompute-modules mlp moe layernorm mla_up_proj
    )
fi

if [[ $SWIGLU_FUSION = false ]]; then
    TRAINING_ARGS+=(
        --no-bias-swiglu-fusion
    )
fi

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --kv-channels 128
    --rotary-scaling-factor 1
    # --mscale 1.0
    # --mscale-all-dim 1.0
    --rotary-base 1000000
    # --beta-fast 1.0
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
)

LEARNING_RATE_ARGS=(
    --lr 3.48e-5
    # --lr-decay-style WSD
    # 236
    --lr-decay-style cosine
    # --lr-wsd-decay-samples  $TRAIN_SAMPLES
    # --lr-wsd-decay-style cosine
    # 236
    --lr-wsd-decay-style exponential
    # --lr-warmup-iters 70
    --lr-warmup-samples ${WARMUP_SAMPLES}
    # --min-lr 2.2e-05
    # 236
    --min-lr 3.48e-05
    --initial-loss-scale 4294967296
    --min-loss-scale 1.0
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP_SIZE
	--pipeline-model-parallel-size $PP_SIZE
    # 236
    --context-parallel-size $CP_SIZE
    --cp-comm-type a2a

    --tp-only-amax-red
    # --num-virtual-stages-per-pipeline-rank 2
)

if [[ ${PP_LAYOUT:-none} != none ]]; then
    MODEL_PARALLEL_ARGS+=(
        --pipeline-model-parallel-layout ${PP_LAYOUT}
    )
else
    MODEL_PARALLEL_ARGS+=(
        --decoder-first-pipeline-num-layers ${FIRST_STAGE}
        --decoder-last-pipeline-num-layers ${LAST_STAGE}
    )
fi

MIXED_PRECISION_ARGS=(
    --bf16
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    # --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer #NullTokenizer
    # --vocab-size 163840
    --tokenizer-model ${TOKENIZED_MODEL}
    --data-cache-path $DATA_CACHE_PATH
    --split 100,0,0
    # --distributed-timeout-minutes 1200
    --distributed-timeout-minutes 10
    --num-dataset-builder-threads 16
    --num-workers 2
    # --no-mmap-bin-files
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    # --save-interval 10000
    #--save-interval 1000
    #--eval-interval 1
    #--save $CHECKPOINT_PATH
    --load $CHECKPOINT_LOAD_PATH
    --eval-iters 0
    --tensorboard-dir $TB_PATH
    --ckpt-format torch_dist
    # --async-save
    --no-create-attention-mask-in-dataloader
    --ckpt-fully-parallel-load
    --use-checkpoint-opt_param-scheduler
    --auto-detect-ckpt-format
    # --no-barrier-with-level-1-timing
    --ckpt-assume-constant-structure
    # --ckpt-load-use-cpu-staging
)

AUX_LOSS_COEF=${AUX_LOSS_COEF:-0.001}
ROUTER_WARM_STEPS=${ROUTER_WARM_STEPS:-0}

NUM_EXPERTS=${NUM_EXPERTS:-160}
EXPERT_SIZE=${EXPERT_SIZE:-1536}
FIRST_K_DENSE=${FIRST_K_DENSE:-1}
MOE_ROUTER_LOAD_BALANCING_TYPE=${MOE_ROUTER_LOAD_BALANCING_TYPE:-aux_loss}
MOE_ROUTER_SCORE_FUNCTION=${MOE_ROUTER_SCORE_FUNCTION:-softmax}

NUM_LAYERS=$(echo "${MODEL_ARGS[@]}" | grep -oP '(?<=--num-layers )\d+')
NUM_LAYERS_MINUS_ONE=$((NUM_LAYERS - $FIRST_K_DENSE))
MOE_LAYER_FREQ="([0]*$FIRST_K_DENSE+[1]*${NUM_LAYERS_MINUS_ONE})*1"
MOE_ARGS=(
    --num-experts $NUM_EXPERTS # 384
    --expert-model-parallel-size $EP_SIZE
    # --moe-router-num-groups 8
    # --moe-router-group-topk 4
    --moe-router-topk 6
    --moe-router-score-function $MOE_ROUTER_SCORE_FUNCTION
    --moe-router-pre-softmax
    --moe-z-loss-coeff 0.0001
    #--moe-router-topk-scaling-factor 2.6539
    --moe-expert-capacity-factor 2.0
    --moe-ffn-hidden-size $EXPERT_SIZE #2048
    --moe-shared-expert-intermediate-size $((${EXPERT_SIZE} * 2 ))
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-grouped-gemm
    # --moe-router-enable-expert-bias
    # --moe-router-bias-update-rate 1e-3
    --router-prob-var-mointor-freq 10
    --router-logit-var-mointor-freq 10
    --router-maxvio-mointor-freq 10
    --moe-router-dtype fp32
    # --moe-router-force-load-balancing
    # --overlap-moe-expert-parallel-comm
    # --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    # --moe-enable-deepep
    # --moe-token-dispatcher-type flex
    # --rotary_seq_len_interpolation_factor 1
    # --router_logit_var_mointor_freq 10
    # --router_maxvio_mointor_freq 10
    # --router_prob_var_mointor_freq 10

    --moe-router-load-balancing-type $MOE_ROUTER_LOAD_BALANCING_TYPE
    --moe-aux-loss-coeff $AUX_LOSS_COEF
)

DEEP_NUM_SM=${DEEP_NUM_SM:-20}
if [ $TOKEN_DISPATCHER = alltoall ]; then
    MOE_ARGS+=(
        --moe-token-dispatcher-type alltoall
    )
elif [ $TOKEN_DISPATCHER = deepep ]; then
    MOE_ARGS+=(
        --moe-token-dispatcher-type flex
        --moe-enable-deepep
        --moe-deepep-num-sms $DEEP_NUM_SM
    )
fi

if [ $FORCE_LB = true ]; then
    MOE_ARGS+=(--moe-router-force-load-balancing)
fi



TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    # --fp8-format e4m3
    # --fp8-param-gather
    # --fp8-recipe mxfp8
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
)

if [ $USE_TOPK_ROUTER_FUSION = true ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --moe-router-fusion
    )
fi

if [ $SE_OVERLAP = true ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --moe-shared-expert-overlap
    )
fi

if [ $FP8 = true ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --fp8-format e4m3
        --fp8-param-gather
        --fp8-recipe mxfp8
    )
fi

if [ $PAO = on ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --use-precision-aware-optimizer
    )
elif [ $PAO = moments ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --use-precision-aware-optimizer
        --exp-avg-dtype fp16
        --exp-avg-sq-dtype fp16
    )
elif [ $PAO = grads ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --use-precision-aware-optimizer
        --main-grads-dtype bf16
    )
elif [ $PAO = params ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --use-precision-aware-optimizer
        --main-params-dtype fp16
    )
elif [ $PAO = all ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --use-precision-aware-optimizer
        --exp-avg-dtype fp16
        --exp-avg-sq-dtype fp16-
        --main-grads-dtype bf16
        --main-params-dtype fp16
    )
else
    MIXED_PRECISION_ARGS+=(
        --accumulate-allreduce-grads-in-fp32
    )
fi

OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-false}
if [ $OFFLOAD_OPTIMIZER = true ]; then
    NEW_ARGS+=(
        --use-precision-aware-optimizer
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
        --use-torch-optimizer-for-cpu-offload
    )
fi

VPP=${VPP:-off}
if [ $VPP = on ]; then
    NEW_ARGS+=(
        --num-virtual-stages-per-pipeline-rank 2
    )
fi

MULTI_TOKEN_PREDICTION_ARGS=(
    # --use-multi-token-prediction
    # --mtp-coeff 1e-4
    # --mtp-num-layers ${MTP}
    # --mtp-loss-scaling-factor 0.3
)

if [ ${CPT_TASK:-false} = true ]; then
    EVAL_AND_LOGGING_ARGS+=(
        --no-load-optim
        --no-load-rng
    )
fi

unset MLFLOW_TRACKING_URI
unset MCCL_IB_HCA
{
    echo "=================================================="
    echo "[$(date '+%F %T')] Hostname: $(hostname)"
    echo "ip=${MY_IP}"
    echo "node_rank=${NODE_RANK}"
    echo "=================================================="
} | tee -a ${LOG_FILE}
#torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_kimi_mtp.py \
#        ${MODEL_ARGS[@]} \
#        ${MULTI_TOKEN_PREDICTION_ARGS[@]} \
#        ${TRAINING_ARGS[@]} \
#        ${REGULARIZATION_ARGS[@]} \
#        ${LEARNING_RATE_ARGS[@]} \
#        ${MODEL_PARALLEL_ARGS[@]} \
#        ${MIXED_PRECISION_ARGS[@]} \
#        ${DATA_ARGS[@]} \
#        ${MOE_ARGS[@]} \
#        ${MLA_ARGS[@]} \
#        ${EVAL_AND_LOGGING_ARGS[@]} \
#        ${NEW_ARGS[@]} \
#        ${TRANSFORMER_ENGINE_ARGS[@]} 2>&1 | tee -a ${LOG_FILE}
#/mnt/zj-data/data/xutk/work/rdma_counter_collector/bin/rdma_counter_ctl.sh stop
#/mnt/zj-data/data/xutk/work/mtlink_counter_collector/bin/mtlink_counter_ctl.sh stop
#/mnt/zj-data/data/xutk/work/rdma_counter_collector/bin/rdma_counter_ctl.sh convert
#/mnt/zj-data/data/xutk/work/mtlink_counter_collector/bin/mtlink_counter_ctl.sh convert
torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_kimi_mtp.py \
        ${MODEL_ARGS[@]} \
        ${MULTI_TOKEN_PREDICTION_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]} \
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${MLA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${NEW_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]} 2>&1 | tee -a ${LOG_FILE}

TORCHRUN_EXIT=${PIPESTATUS[0]}

echo "torchrun exit code: ${TORCHRUN_EXIT}"

# 无论torchrun成功失败，都继续执行
/mnt/zj-data/data/xutk/work/rdma_counter_collector/bin/rdma_counter_ctl.sh stop
/mnt/zj-data/data/xutk/work/mtlink_counter_collector/bin/mtlink_counter_ctl.sh stop
/mnt/zj-data/data/xutk/work/rdma_counter_collector/bin/rdma_counter_ctl.sh convert
/mnt/zj-data/data/xutk/work/mtlink_counter_collector/bin/mtlink_counter_ctl.sh convert

exit ${TORCHRUN_EXIT}
set +x
