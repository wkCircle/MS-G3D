# current train shape: (N,C,T,V,M) = (94477*.95, 3, 126, 65, 1)


### the main code to train/validate model 
# python main.py --config ./config/kinetics-skeleton/train_joint.yaml
python main.py --config ./config/asl/train_joint.yaml --checkpoint <path/to/file>
# when num_point=65, batch_size=32, forward_batch_size=16 => 40 mins/epoch or 1.2batch(32)/s
# tra/val mean batch(32) loss and acc. ep1:5.1/4.3/4.6%; ep2:4.3/3.6/15.2% ep3:3.6/3.2/24.5%

### tensorboard to visualize train/val log files: the url will be "localhost:6006"
tensorboard --logdir=./work_dir/asl/msg3d_joint/trainlogs

