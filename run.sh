export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
declare -A map_tmax_new
map_tmax_new=(
    ["3m"]=1000000
    ["5m"]=1000000
    ["7m"]=1000000
    ["8m"]=1000000
    ["2s3z"]=1050000
    ["3s5z"]=2050000
    ["1c3s5z"]=2050000
    ["2c_vs_64zg"]=2050000
    ["3s_vs_3z"]=2050000
    ["3s_vs_4z"]=2050000
    ["3s_vs_5z"]=2050000
    ["5m_vs_6m"]=2050000
    ["8m_vs_9m"]=2050000
    ["10m_vs_11m"]=2050000
    ["MMM2"]=3050000
    ["3s5z_vs_3s6z"]=3050000
    ["6h_vs_8z"]=5050000
    ["corridor"]=5050000
)

for map_name in  "3s_vs_3z" 
do
    t_max=${map_tmax_new[$map_name]}

    # 在不同 GPU 上同時執行
    export CUDA_VISIBLE_DEVICES=0
    python3 src/main.py --config=updeept_qmix --env-config=sc2 with env_args.map_name=$map_name  t_max=$t_max device_name=0  

    # export CUDA_VISIBLE_DEVICES=1
    # python3 src/main.py --config=updeept_qmix --env-config=sc2 with env_args.map_name=$map_name  t_max=$t_max device_name=1 > /dev/null & 

    wait
done