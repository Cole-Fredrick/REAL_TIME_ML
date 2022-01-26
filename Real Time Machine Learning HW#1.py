#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import models
dir(models)


# In[2]:


################################################################################################################################
# Problem 1
################################################################################################################################


# In[3]:


resnet = models.resnet101(pretrained=True)


# In[4]:


resnet


# In[5]:


from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


# In[6]:


from PIL import Image
img = Image.open("dog1.jpg")
img2 = Image.open("cat2.jpg")
img3 = Image.open("chicken3.jpg")
img4 = Image.open("horse4.jpg")
img5 = Image.open("duck5.jpg")


# In[7]:


img_t = preprocess(img)
img_t2 = preprocess(img2)
img_t3 = preprocess(img3)
img_t4 = preprocess(img4)
img_t5 = preprocess(img5)


# In[8]:


import torch
batch_t = torch.unsqueeze(img_t, 0)
batch_t2 = torch.unsqueeze(img_t2, 0)
batch_t3 = torch.unsqueeze(img_t3, 0)
batch_t4 = torch.unsqueeze(img_t4, 0)
batch_t5 = torch.unsqueeze(img_t5, 0)


# In[9]:


resnet.eval()


# In[10]:


out = resnet(batch_t)
out2 = resnet(batch_t2)
out3 = resnet(batch_t3)
out4 = resnet(batch_t4)
out5 = resnet(batch_t5)


# In[11]:


with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


# In[12]:


_, index = torch.max(out, 1)
_, index2 = torch.max(out2, 1)
_, index3 = torch.max(out3, 1)
_, index4 = torch.max(out4, 1)
_, index5 = torch.max(out5, 1)


# In[13]:


percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
percentage2 = torch.nn.functional.softmax(out2, dim=1)[0] * 100
percentage3 = torch.nn.functional.softmax(out3, dim=1)[0] * 100
percentage4 = torch.nn.functional.softmax(out4, dim=1)[0] * 100
percentage5 = torch.nn.functional.softmax(out5, dim=1)[0] * 100


# In[14]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[15]:


_, indices2 = torch.sort(out2, descending=True)
[(labels[idx], percentage2[idx].item()) for idx in indices2[0][:5]]


# In[16]:


_, indices3 = torch.sort(out3, descending=True)
[(labels[idx], percentage3[idx].item()) for idx in indices3[0][:5]]


# In[17]:


_, indices4 = torch.sort(out4, descending=True)
[(labels[idx], percentage4[idx].item()) for idx in indices4[0][:5]]


# In[18]:


_, indices5 = torch.sort(out5, descending=True)
[(labels[idx], percentage5[idx].item()) for idx in indices5[0][:5]]


# In[19]:


################################################################################################################################
# Problem 2
################################################################################################################################


# In[20]:


import torch
import torch.nn as nn

class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)


# In[21]:


netG = ResNetGenerator()


# In[22]:


model_path = 'horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)


# In[23]:


netG.eval()


# In[24]:


from PIL import Image
from torchvision import transforms


# In[25]:


preprocess = transforms.Compose([transforms.Resize(256),
transforms.ToTensor()])


# In[26]:


img = Image.open("horse1.jpg")
img2 = Image.open("horse2.jpg")
img3 = Image.open("horse3.jpg")
img4 = Image.open("horse4.jpg")
img5 = Image.open("horse5.jpg")


# In[27]:


img_t = preprocess(img)
img_t2 = preprocess(img2)
img_t3 = preprocess(img3)
img_t4 = preprocess(img4)
img_t5 = preprocess(img5)


# In[28]:


batch_t = torch.unsqueeze(img_t, 0)
batch_t2 = torch.unsqueeze(img_t2, 0)
batch_t3 = torch.unsqueeze(img_t3, 0)
batch_t4 = torch.unsqueeze(img_t4, 0)
batch_t5 = torch.unsqueeze(img_t5, 0)


# In[29]:


batch_out = netG(batch_t)
batch_out2 = netG(batch_t2)
batch_out3 = netG(batch_t3)
batch_out4 = netG(batch_t4)
batch_out5 = netG(batch_t5)


# In[30]:


out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
# out_img.save('../data/p1ch2/zebra.jpg')
out_img


# In[31]:


out_t2 = (batch_out2.data.squeeze() + 1.0) / 2.0
out_img2 = transforms.ToPILImage()(out_t2)
# out_img.save('../data/p1ch2/zebra.jpg')
out_img2


# In[32]:


out_t3 = (batch_out3.data.squeeze() + 1.0) / 2.0
out_img3 = transforms.ToPILImage()(out_t3)
# out_img.save('../data/p1ch2/zebra.jpg')
out_img3


# In[33]:


out_t4 = (batch_out4.data.squeeze() + 1.0) / 2.0
out_img4 = transforms.ToPILImage()(out_t4)
# out_img.save('../data/p1ch2/zebra.jpg')
out_img4


# In[34]:


out_t5 = (batch_out5.data.squeeze() + 1.0) / 2.0
out_img5 = transforms.ToPILImage()(out_t5)
# out_img.save('../data/p1ch2/zebra.jpg')
out_img5


# In[35]:


################################################################################################################################
# Problem 3
################################################################################################################################


# In[36]:


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = models.resnet101()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[37]:


with torch.cuda.device(0):
  net = netG
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[38]:


################################################################################################################################
# Problem 4
################################################################################################################################


# In[39]:


import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


# In[40]:


from PIL import Image
img = Image.open("dog1.jpg")
img2 = Image.open("cat2.jpg")
img3 = Image.open("chicken3.jpg")
img4 = Image.open("horse4.jpg")
img5 = Image.open("duck5.jpg")


# In[41]:


from torchvision import transforms
input_image = img
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[42]:


with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[43]:


from torchvision import transforms
input_image = img2
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[44]:


with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[45]:


from torchvision import transforms
input_image = img3
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[46]:


with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[47]:


from torchvision import transforms
input_image = img4
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[48]:


with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[49]:


from torchvision import transforms
input_image = img5
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[50]:


# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[51]:


with torch.cuda.device(0):
  net = models.MobileNetV2()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:




