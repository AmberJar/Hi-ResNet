import torch

# # Model
print("Load model ............")
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
print("[device]", device)
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
# Load checkpoint
checkpoint = torch.load(args.model, map_location=device)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']


# If during training, we used data parallel
if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
    # for gpu inference, use data parallel
    if "cuda" in device.type:
        print("HERE")
        # model = torch.nn.DataParallel(model)
    # else:
        # for cpu inference, remove module
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint = new_state_dict
# load
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval() # freeze BN layer and backward memory

print("Load model complete.>>>")
x = torch.rand(1, 3, 512, 512).cuda()  # dummy data
traced_script_module = torch.jit.trace(model, x)
traced_script_module.save(r"./weights/model.pt")
print("torch.jit save model complete.>>>")

# print("Load model ... >>>")
# availble_gpus = list(range(torch.cuda.device_count()))
# device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
# model = torch.jit.load(r"./weights/model.pt")
# model.to(device)
# model.eval()
# print("Load model complete.>>>")