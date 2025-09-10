## v1.7.0 (2025-09-11)

### Refactor

- move None check outside function

## v1.6.0 (2025-09-10)

### Feat

- **gpu**: add CUDA_VISIBLE_DEVICES support

### Fix

- the config test cannot works

## v1.5.0 (2025-09-07)

### Feat

- add args of friendly_min to avoid OOM from previous job's final test/cleanup

## v1.4.0 (2025-09-03)

### Feat

- add spinner in the processing of snatching gpus

## v1.3.2 (2025-09-03)

### Fix

- The snatched gpu index is error in Notifications
- counts error of snatchered gpu if worker not works

### Refactor

- simulation task for gpu

## v1.3.0 (2025-09-02)

### Feat

- to simulate fluctuating GPU utilization

## v1.2.2 (2025-09-02)

### Fix

- change the threshold ratio of free/total memory (0.10 -> 0.85)

## v1.2.1 (2025-09-02)

## v1.2.0 (2025-09-02)

## v1.1.1 (2025-09-02)

### Fix

- only run ci in main branch & remove CHANGE LOG in publish

## v1.1.0 (2025-09-02)

### Feat

- add spinner for waiting time
- toy version
- process email_pwd
- add config module

### Fix

- can not use nvidia-smi in github actions
- the index name of torch
- response error from email
