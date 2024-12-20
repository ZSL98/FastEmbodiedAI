#!/bin/bash

# 检查是否传递了模型名称参数
if [ -z "$1" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

MODEL_NAME="$1"

CONFIG_FILES=("config_seq.yaml" "config_ours.yaml")

if [ "$MODEL_NAME" == "openvla" ]; then
    # 循环遍历每个配置文件，并修改 decode_len
    for file in "${CONFIG_FILES[@]}"
    do
    if [ -f "$file" ]; then
        sed -i "s/decode_len: [0-9]\+/decode_len: 6/" $file
        echo "Updated decode_len to 6 in $file"
    else
        echo "Warning: $file not found, cannot update decode_len."
    fi
    done
elif [ "$MODEL_NAME" == "llava" ]; then
    # 循环遍历每个配置文件，并修改 decode_len
    for file in "${CONFIG_FILES[@]}"
    do
    if [ -f "$file" ]; then
        sed -i "s/decode_len: [0-9]\+/decode_len: 10/" $file
        echo "Updated decode_len to 10 in $file"
    else
        echo "Warning: $file not found, cannot update decode_len."
    fi
    done
elif [ "$MODEL_NAME" == "diffusion_cnn" ]; then
    # 循环遍历每个配置文件，并修改 diffusion_step 为 40
    for file in "${CONFIG_FILES[@]}"
    do
    if [ -f "$file" ]; then
        sed -i "s/diffusion_step: [0-9]\+/diffusion_step: 40/" $file
        sed -i "s/diffusion_stage_num: [0-9]\+/diffusion_stage_num: 4/" $file
        echo "Updated diffusion_step to 40 in $file"
    else
        echo "Warning: $file not found, cannot update diffusion_step."
    fi
    done
elif [ "$MODEL_NAME" == "diffusion_transformer" ]; then
    # 循环遍历每个配置文件，并修改 diffusion_step 为 100
    for file in "${CONFIG_FILES[@]}"
    do
    if [ -f "$file" ]; then
        sed -i "s/diffusion_step: [0-9]\+/diffusion_step: 100/" $file
        sed -i "s/diffusion_stage_num: [0-9]\+/diffusion_stage_num: 5/" $file
        echo "Updated diffusion_step to 100 in $file"
    else
        echo "Warning: $file not found, cannot update diffusion_step."
    fi
    done
else
    echo "Unknown model: $MODEL_NAME. No modifications were made."
fi

# 如果 MODEL_NAME 是 openvla，则进行进一步的修改
if [ "$MODEL_NAME" == "openvla" ]; then
    for tokens in 7 14 21 28 35 42 49 56
    do
        # 修改 inference.py 中的 input_seq_len
        sed -i "s/self.input_seq_len = args.input_seq_len + 512 + [0-9]\+/self.input_seq_len = args.input_seq_len + 512 + $((tokens - 1))/" ./openvla_inference/inference.py

        # 循环遍历每个配置文件，并修改 decode_len
        for file in "${CONFIG_FILES[@]}"
        do
        if [ -f "$file" ]; then
            sed -i "s/decode_len: [0-9]\+/decode_len: $((tokens - 1))/" $file
            echo "Updated decode_len to '$((tokens - 1))' in $file"
        else
            echo "Warning: $file not found, cannot update decode_len."
        fi
        done

        # 更新模型名称
        for file in "${CONFIG_FILES[@]}"
        do
        if [ -f "$file" ]; then
            sed -i "s/model: \".*\"/model: \"$MODEL_NAME\"/" $file
            echo "Updated model to '$MODEL_NAME' in $file"
        else
            echo "Warning: $file not found, cannot update model."
        fi
        done

        # 记录日志文件
        LOG_FILE="${MODEL_NAME}_${tokens}_seq.log"
        
        # 串行运行 5 次命令 (Seq 部分)
        for i in {1..5}
        do
        echo "Running iteration $i with $tokens tokens..." >> $LOG_FILE
        python sche_plan.py --config ./config_seq.yaml >> $LOG_FILE 2>&1
        echo "Iteration $i completed." >> $LOG_FILE
        echo "--------------------------------------" >> $LOG_FILE
        done

        echo "All iterations are complete. Output saved to $LOG_FILE." >> $LOG_FILE

        # 计算 Query duration 的平均值
        average_query_duration=$(awk '/Query duration:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
        average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration / 1000" | bc))

        # 输出平均值到日志文件
        echo "Average Query Duration: $average_query_duration ms ($average_query_duration_seconds seconds)" >> $LOG_FILE

        if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
            throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
            echo "Throughput: $throughput requests/second" >> $LOG_FILE
        else
            echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
        fi

        # Ours 运行 5 次命令 (Ours 部分)
        LOG_FILE="${MODEL_NAME}_${tokens}_ours.log"

        for i in {1..5}
        do
        echo "Running iteration $i with $tokens tokens..." >> $LOG_FILE
        python sche_plan.py --config ./config_ours.yaml >> $LOG_FILE 2>&1
        echo "Iteration $i completed." >> $LOG_FILE
        echo "--------------------------------------" >> $LOG_FILE
        done

        echo "All iterations are complete. Output saved to $LOG_FILE." >> $LOG_FILE

        # 计算 Frame interval 的平均值
        average_query_duration=$(awk '/Frame interval:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
        average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration" | bc))

        # 输出平均值到日志文件
        echo "Average Query Duration: $average_query_duration s ($average_query_duration_seconds seconds)" >> $LOG_FILE

        if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
            throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
            echo "Throughput: $throughput requests/second" >> $LOG_FILE
        else
            echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
        fi
    done
# 如果 MODEL_NAME 是 llava，则进行进一步的修改
elif [ "$MODEL_NAME" == "llava" ]; then
    for tokens in 7 14 21 28 35 42 49 56
    do
        # 修改 inference.py 中的 input_seq_len
        sed -i "s/self.input_seq_len = args.input_seq_len + 576 + [0-9]\+/self.input_seq_len = args.input_seq_len + 576 + $((tokens - 1))/" ./llava_inference/inference.py

        # 循环遍历每个配置文件，并修改 decode_len
        for file in "${CONFIG_FILES[@]}"
        do
        if [ -f "$file" ]; then
            sed -i "s/decode_len: [0-9]\+/decode_len: $((tokens - 1))/" $file
            echo "Updated decode_len to '$((tokens - 1))' in $file"
        else
            echo "Warning: $file not found, cannot update decode_len."
        fi
        done

        # 更新模型名称
        for file in "${CONFIG_FILES[@]}"
        do
        if [ -f "$file" ]; then
            sed -i "s/model: \".*\"/model: \"$MODEL_NAME\"/" $file
            echo "Updated model to '$MODEL_NAME' in $file"
        else
            echo "Warning: $file not found, cannot update model."
        fi
        done

        # 记录日志文件
        LOG_FILE="${MODEL_NAME}_${tokens}_seq.log"
        
        # 串行运行 5 次命令 (Seq 部分)
        for i in {1..5}
        do
        echo "Running iteration $i with $tokens tokens..." >> $LOG_FILE
        python sche_plan.py --config ./config_seq.yaml >> $LOG_FILE 2>&1
        echo "Iteration $i completed." >> $LOG_FILE
        echo "--------------------------------------" >> $LOG_FILE
        done

        echo "All iterations are complete. Output saved to $LOG_FILE." >> $LOG_FILE

        # 计算 Query duration 的平均值
        average_query_duration=$(awk '/Query duration:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
        average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration / 1000" | bc))

        # 输出平均值到日志文件
        echo "Average Query Duration: $average_query_duration ms ($average_query_duration_seconds seconds)" >> $LOG_FILE

        if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
            throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
            echo "Throughput: $throughput requests/second" >> $LOG_FILE
        else
            echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
        fi

        # Ours 运行 5 次命令 (Ours 部分)
        LOG_FILE="${MODEL_NAME}_${tokens}_ours.log"

        for i in {1..5}
        do
        echo "Running iteration $i with $tokens tokens..." >> $LOG_FILE
        python sche_plan.py --config ./config_ours.yaml >> $LOG_FILE 2>&1
        echo "Iteration $i completed." >> $LOG_FILE
        echo "--------------------------------------" >> $LOG_FILE
        done

        echo "All iterations are complete. Output saved to $LOG_FILE." >> $LOG_FILE

        # 计算 Frame interval 的平均值
        average_query_duration=$(awk '/Frame interval:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
        average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration" | bc))

        # 输出平均值到日志文件
        echo "Average Query Duration: $average_query_duration s ($average_query_duration_seconds seconds)" >> $LOG_FILE

        if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
            throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
            echo "Throughput: $throughput requests/second" >> $LOG_FILE
        else
            echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
        fi
    done
# 如果 MODEL_NAME 是 diffusion_cnn，则进行进一步的修改
elif [[ "$MODEL_NAME" == "diffusion_cnn" || "$MODEL_NAME" == "diffusion_transformer" ]]; then
    for diffusion_steps in 20 40 60 80 100 120 140 160
    do
        # 修改每个配置文件中的 diffusion_step
        for file in "${CONFIG_FILES[@]}"
        do
        if [ -f "$file" ]; then
            sed -i "s/diffusion_step: [0-9]\+/diffusion_step: $diffusion_steps/" $file
            echo "Updated diffusion_step to '$diffusion_steps' in $file"
        else
            echo "Warning: $file not found, cannot update diffusion_step."
        fi
        done

        # 更新模型名称
        for file in "${CONFIG_FILES[@]}"
        do
        if [ -f "$file" ]; then
            sed -i "s/model: \".*\"/model: \"$MODEL_NAME\"/" $file
            echo "Updated model to '$MODEL_NAME' in $file"
        else
            echo "Warning: $file not found, cannot update model."
        fi
        done

        # 记录日志文件
        LOG_FILE="${MODEL_NAME}_${diffusion_steps}_seq.log"
        
        # 串行运行 5 次命令 (Seq 部分)
        for i in {1..5}
        do
        echo "Running iteration $i with $diffusion_steps diffusion steps..." >> $LOG_FILE
        python sche_plan.py --config ./config_seq.yaml >> $LOG_FILE 2>&1
        echo "Iteration $i completed." >> $LOG_FILE
        echo "--------------------------------------" >> $LOG_FILE
        done

        echo "All iterations are complete. Output saved to $LOG_FILE." >> $LOG_FILE

        # 计算 Query duration 的平均值
        average_query_duration=$(awk '/Query duration:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
        average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration / 1000" | bc))

        # 输出平均值到日志文件
        echo "Average Query Duration: $average_query_duration ms ($average_query_duration_seconds seconds)" >> $LOG_FILE

        if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
            throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
            echo "Throughput: $throughput requests/second" >> $LOG_FILE
        else
            echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
        fi

        # Ours 运行 5 次命令 (Ours 部分)
        LOG_FILE="${MODEL_NAME}_${diffusion_steps}_ours.log"

        for i in {1..5}
        do
        echo "Running iteration $i with $diffusion_steps diffusion steps..." >> $LOG_FILE
        python sche_plan.py --config ./config_ours.yaml >> $LOG_FILE 2>&1
        echo "Iteration $i completed." >> $LOG_FILE
        echo "--------------------------------------" >> $LOG_FILE
        done

        echo "All iterations are complete. Output saved to $LOG_FILE." >> $LOG_FILE

        # 计算 Frame interval 的平均值
        average_query_duration=$(awk '/Frame interval:/ {duration_sum += $3; count++} END {print duration_sum/count}' $LOG_FILE)
        average_query_duration_seconds=$(printf "%.8f" $(echo "scale=8; $average_query_duration" | bc))

        # 输出平均值到日志文件
        echo "Average Query Duration: $average_query_duration s ($average_query_duration_seconds seconds)" >> $LOG_FILE

        if (( $(echo "$average_query_duration_seconds > 0" | bc -l) )); then
            throughput=$(printf "%.8f" $(echo "scale=8; 1 / $average_query_duration_seconds" | bc))
            echo "Throughput: $throughput requests/second" >> $LOG_FILE
        else
            echo "Throughput: Infinite (duration zero)" >> $LOG_FILE
        fi
    done
else
    echo "Unkown model name"
fi