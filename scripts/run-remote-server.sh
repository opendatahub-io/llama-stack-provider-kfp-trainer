#!/bin/sh

. .venv/bin/activate

tmpdir=$(mktemp -d)
trap 'rm -rf $tmpdir' EXIT

# patch run.yaml
cp run.yaml $tmpdir
sed -i "s/local/remote/g" $tmpdir/run.yaml

llama stack run $tmpdir/run.yaml
