mkdir -p results/trainlogs

echo "开始测试gchr"
for seed in 100 200 300 400 500
do
    CUDA_VISIBLE_DEVICES=6 python main.py --env-name=FetchReach --lr-actor 0.001 --lr-critic 0.001 --n-epochs 20  --agent gchr --negative-reward --seed $seed  --cuda
    CUDA_VISIBLE_DEVICES=6 python main.py --env-name=FetchPick --lr-actor 0.001 --lr-critic 0.001 --n-epochs 20  --agent gchr --negative-reward --seed $seed  --cuda
    CUDA_VISIBLE_DEVICES=6 python main.py --env-name=FetchSlide --lr-actor 0.001 --lr-critic 0.001 --n-epochs 20  --agent gchr --negative-reward --seed $seed  --cuda
    CUDA_VISIBLE_DEVICES=6 python main.py --env-name=FetchPush --lr-actor 0.001 --lr-critic 0.001 --n-epochs 20  --agent gchr --negative-reward --seed $seed  --cuda
    CUDA_VISIBLE_DEVICES=6 python main.py --env-name=HandReach --lr-actor 0.001 --lr-critic 0.001 --n-epochs 50 --agent gchr --negative-reward --seed $seed  --cuda
    CUDA_VISIBLE_DEVICES=6 python main.py --env-name=HandManipulateBlockRotateZ --lr-actor 0.001 --lr-critic 0.001 --n-epochs 50  --agent gchr --negative-reward --seed $seed  --cuda
    CUDA_VISIBLE_DEVICES=6 python main.py --env-name=HandManipulateBlockRotateParallel --lr-actor 0.001 --lr-critic 0.001 --n-epochs 50 --agent gchr --negative-reward --seed $seed  --cuda
    CUDA_VISIBLE_DEVICES=6 python main.py --env-name=HandManipulateBlockRotateXYZ --lr-actor 0.001 --lr-critic 0.001 --n-epochs 50 --agent gchr --negative-reward --seed $seed  --cuda
done
echo "结束测试gchr"
