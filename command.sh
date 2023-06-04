# current train shape: (N,C,T,V,M) = (94477*.95, 3, 126, 65, 1)


### the main code to train/validate model 
# python main.py --config ./config/kinetics-skeleton/train_joint.yaml
python main.py --config ./config/asl/train_joint.yaml --start-epoch <last-saved-epoch-number> --weights <path/to/file> --checkpoint <path/to/file>


### tensorboard to visualize train/val log files: the url will be "localhost:6006"
tensorboard --logdir=./work_dir/asl/msg3d_joint/trainlogs

