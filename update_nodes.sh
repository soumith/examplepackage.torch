echo "copying train-on-cifar files to End, Fury, and Sorrow worker nodes"

scp -r /home/ubuntu/demos-master/AutomatedParallelization.torch/ ubuntu@10.0.0.220:/home/ubuntu/demos-master/ > log.txt
echo "[1/7] Copy to End complete"
scp -r /home/ubuntu/demos-master/AutomatedParallelization.torch/ ubuntu@10.0.0.164:/home/ubuntu/demos-master/ >> log.txt
echo "[2/7] Copy to Sorrow complete"
scp -r /home/ubuntu/demos-master/AutomatedParallelization.torch/ ubuntu@10.0.0.101:/home/ubuntu/demos-master/ >> log.txt
echo "[3/7] Copy to Fury complete"

ssh ubuntu@10.0.0.220 '/home/ubuntu/demos-master/AutomatedParallelization.torch/reinstall.sh' >> log.txt
echo "[4/7] Reinstall on End complete"
ssh ubuntu@10.0.0.164 '/home/ubuntu/demos-master/AutomatedParallelization.torch/reinstall.sh' >> log.txt
echo "[5/7] Reinstall on Sorrow complete"
ssh ubuntu@10.0.0.101 '/home/ubuntu/demos-master/AutomatedParallelization.torch/reinstall.sh' >> log.txt
echo "[6/7] Reinstall on Fury complete"

./reinstall.sh &>> log.txt
echo "[7/7] Reinstall on Pain complete"
