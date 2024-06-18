import multiprocessing

import torch

from .io import load_checkpoint


def worker_func(model_cls, model_kwargs, checkpoint, dataset, data_func,
                gpu_id, idx_queue, result_queue):
    model = model_cls(**model_kwargs)
    load_checkpoint(model, checkpoint, map_location='cpu')
    torch.cuda.set_device(gpu_id)
    model.cuda()
    model.eval()
    with torch.no_grad():
        while True:
            idx = idx_queue.get()
            data = dataset[idx]
            result = model(**data_func(data, gpu_id))
            result_queue.put((idx, result))


def parallel_test(model_cls,
                  model_kwargs,
                  checkpoint,
                  dataset,
                  data_func,
                  gpus,
                  worker_per_gpu=1):
    ctx = multiprocessing.get_context('spawn')
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = len(gpus) * worker_per_gpu
    workers = [
        ctx.Process(
            target=worker_func,
            args=(model_cls, model_kwargs, checkpoint, dataset, data_func,
                  gpus[i % len(gpus)], idx_queue, result_queue))
        for i in range(num_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    for i in range(len(dataset)):
        idx_queue.put(i)

    results = [None for _ in range(len(dataset))]
    import cvbase as cvb
    prog_bar = cvb.ProgressBar(task_num=len(dataset))
    for _ in range(len(dataset)):
        idx, res = result_queue.get()
        results[idx] = res
        prog_bar.update()
    print('\n')
    for worker in workers:
        worker.terminate()

    return results
