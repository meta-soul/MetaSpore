if [ ! -f $(dirname ${BASH_SOURCE[0]})/python-env/requirements.txt ]
then
    $(dirname ${BASH_SOURCE[0]})/env.restore.sh
fi

source $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env/bin/activate
