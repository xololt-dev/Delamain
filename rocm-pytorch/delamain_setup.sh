docker exec -it "$1" sh -c "mkdir /Delamain"
docker cp . "$1":/Delamain
docker exec -it "$1" sh -c "cd /Delamain; bash /Delamain/rocm-pytorch/delamain_cont_setup.sh"

sleep 15m
while true; do
    docker cp "$1":/Delamain/ "$2"
    if [ $? -ne 0 ]; then
        break
    else
        sleep 15m
    fi;
done;
