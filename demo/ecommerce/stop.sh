docker-compose -f init_container/create-database.yml down

source ./env.sh
cd ecommerce_demo || exit
metaspore flow down
