#!/bin/bash

for pretrained in True False
do
    for model in resnet18 resnet34 resnet50 mobeilenet mobilenetv2 mobilenetv3
    do
        for frames in 96 64 32 16 8 4 1
        do
            batch=$((256 / frames))
            batch=$(( batch > 16 ? 16 : batch ))

            cmd="from model.py import ef_module; ef_module(model_name=\"${model}\", frames=${frames}, period=1, pretrained=${pretrained}, batch_size=${batch})"
            python3 -c "${cmd}"
        done
        for period in 2 4 6 8
        do
            batch=$((256 / 64 * period))
            batch=$(( batch > 16 ? 16 : batch ))

            cmd="from model.py import ef_module; ef_module(model_name=\"${model}\", frames=(64 // ${period}), period=${period}, pretrained=${pretrained}, batch_size=${batch})"
            python3 -c "${cmd}"
        done
    done
done

period=2
pretrained=True
for model in resnet18 resnet34 resnet50 mobeilenet mobilenetv2 mobilenetv3
do
    cmd="from model.py import ef_module; ef_module(model_name=\"${model}\", frames=(64 // ${period}), period=${period}, pretrained=${pretrained}, run_test=True)"
    python3 -c "${cmd}"
done

python3 -c "from video.py import ef_module; ef_module(model_name=\"deeplabv3_resnet50\",  save_segmentation=True, pretrained=False)"
ef_module
pretrained=True
model=mobilenetv2
period=2
batch=$((256 / 64 * period))
batch=$(( batch > 16 ? 16 : batch ))
for patients in 16 32 64 128 256 512 1024 2048 4096 7460
do
    cmd="from model.py import ef_module; ef_module(model_name=\"${model}\", frames=(64 // ${period}), period=${period}, pretrained=${pretrained}, batch_size=${batch}, num_epochs=min(50 * (8192 // ${patients}), 200), output=\"output/training_size/video/${patients}\", n_train_patients=${patients})"
    python3 -c "${cmd}"

done

