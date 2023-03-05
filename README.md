## Basic AiXpand processing node capability test

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
docker volume create 
docker run --pull=always --gpus all aixpand/tester
```
