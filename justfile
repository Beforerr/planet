default:
  just --list

env:
  mkdir -p data
  # micromamba create -f environment.yml

env-update:
  micromamba update --file environment.yml