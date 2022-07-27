I am trying to fine-tune wavlm for a downstream task, but when I backprop the loss (cross entropy) I get the below stack trace. It seems to be pointing to some inplace operation being performed in the `pos_conv` layer. Here's a barebones code snippet I used to produce the error:

```
import torch
import torch.nn.functional as F
from WavLM import WavLM, WavLMConfig

torch.autograd.set_detect_anomaly(True)

checkpoint = torch.load('WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])

model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.train()

wav_input_16khz = torch.randn(2,10000)
y_hat, _ = model.extract_features(wav_input_16khz)

loss = F.cross_entropy(y_hat.view(-1, 1024), torch.ones((y_hat.view(-1, 1024).size(0)), dtype=torch.long))
loss.backward()
```

Stack trace:

```
/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/autograd/__init__.py:173: UserWarning: Error detected in ConvolutionBackward0. Traceback of forward call that caused the error:
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/ipykernel_launcher.py", line 16, in <module>
    app.launch_new_instance()
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/traitlets/config/application.py", line 846, in launch_instance
    app.start()
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/ipykernel/kernelapp.py", line 677, in start
    self.io_loop.start()
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/tornado/platform/asyncio.py", line 199, in start
    self.asyncio_loop.run_forever()
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/asyncio/base_events.py", line 601, in run_forever
    self._run_once()
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/asyncio/base_events.py", line 1905, in _run_once
    handle._run()
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/asyncio/events.py", line 80, in _run
    self._context.run(self._callback, *self._args)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/ipykernel/kernelbase.py", line 471, in dispatch_queue
    await self.process_one()
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/ipykernel/kernelbase.py", line 460, in process_one
    await dispatch(*args)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/ipykernel/kernelbase.py", line 367, in dispatch_shell
    await result
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/ipykernel/kernelbase.py", line 662, in execute_request
    reply_content = await reply_content
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/ipykernel/ipkernel.py", line 360, in do_execute
    res = shell.run_cell(code, store_history=store_history, silent=silent)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/ipykernel/zmqshell.py", line 532, in run_cell
    return super().run_cell(*args, **kwargs)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 2880, in run_cell
    result = self._run_cell(
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 2935, in _run_cell
    return runner(coro)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/IPython/core/async_helpers.py", line 129, in _pseudo_sync_runner
    coro.send(None)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 3134, in run_cell_async
    has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 3337, in run_ast_nodes
    if await self.run_code(code, result, async_=asy):
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 3397, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "/tmp/ipykernel_76715/3058532428.py", line 14, in <cell line: 14>
    y_hat, lengths = model.extract_features(wav_input_16khz)
  File "/home/gfrost/projects/penguin/notebooks/WavLM.py", line 364, in extract_features
    x, layer_results = self.encoder(
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/gfrost/projects/penguin/notebooks/WavLM.py", line 565, in forward
    x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)
  File "/home/gfrost/projects/penguin/notebooks/WavLM.py", line 577, in extract_features
    x_conv = self.pos_conv(x.transpose(1, 2))
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/nn/modules/container.py", line 139, in forward
    input = module(input)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1148, in _call_impl
    result = forward_call(*input, **kwargs)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 307, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/home/gfrost/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 303, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
 (Triggered internally at  /opt/conda/conda-bld/pytorch_1656352660876/work/torch/csrc/autograd/python_anomaly_mode.cpp:102.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/home/gfrost/projects/penguin/notebooks/model_tinker.ipynb Cell 33 in <cell line: 17>()
     [14](vscode-notebook-cell://ssh-remote%2Bunpulse3.sun.ac.za/home/gfrost/projects/penguin/notebooks/model_tinker.ipynb#ch0000049vscode-remote?line=13) y_hat, lengths = model.extract_features(wav_input_16khz)
     [16](vscode-notebook-cell://ssh-remote%2Bunpulse3.sun.ac.za/home/gfrost/projects/penguin/notebooks/model_tinker.ipynb#ch0000049vscode-remote?line=15) loss = F.cross_entropy(y_hat.view(-1, 1024), torch.ones((y_hat.view(-1, 1024).size(0)), dtype=torch.long))
---> [17](vscode-notebook-cell://ssh-remote%2Bunpulse3.sun.ac.za/home/gfrost/projects/penguin/notebooks/model_tinker.ipynb#ch0000049vscode-remote?line=16) loss.backward()

File ~/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/_tensor.py:396, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
    387 if has_torch_function_unary(self):
    388     return handle_torch_function(
    389         Tensor.backward,
    390         (self,),
   (...)
    394         create_graph=create_graph,
    395         inputs=inputs)
--> 396 torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)

File ~/anaconda3/envs/penguin/lib/python3.9/site-packages/torch/autograd/__init__.py:173, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    168     retain_graph = create_graph
    170 # The reason we repeat same the comment below is that
    171 # some Python versions print out the first line of a multi-line function
    172 # calls in the traceback and some print out the last line
--> 173 Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    174     tensors, grad_tensors_, retain_graph, create_graph, inputs,
    175     allow_unreachable=True, accumulate_grad=True)

RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [2, 1024, 31]], which is output 0 of AsStridedBackward0, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
```
Incase it's a versioning issue, I'm using `PyTorch 1.12` and `cuda 11.3` (although the above stack trace was on cpu, the same happens with gpu). Any help would be much appreciated, I'm totally lost.
