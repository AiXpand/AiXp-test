# Basic AiXpand processing node capability test

## Docker run
For no-GPU machines run: 
```
docker run --pull=always aixpand/tester
``` 

while for machines with GPU run 
```
docker run --pull=always --gpus all aixpand/tester
```

### Saving data

```
docker volume create tester
docker run --pull=always --gpus all -v tester:/test_app/output aixpand/tester
```

## Non-docker run

After cloning the repo from [https://github.com/AiXpand/AiXp_test.git](https://github.com/AiXpand/AiXp_test.git) make sure you have a fully working anaconda env with `exe_env` specs
```
conda activate <env>
cd <clone_dir>
python test.py
```
