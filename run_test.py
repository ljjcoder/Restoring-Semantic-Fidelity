import torch
from flux_pipeline import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from utils import set_seeds
import pdb
import argparse
import os
import glob
import time
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    if v.lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--iter_id', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--use_tvsda_attention', type=str2bool, default=False)
parser.add_argument('--use_npa_attention', type=str2bool, default=False)
parser.add_argument('--use_local_attention', type=str2bool, default=False)

parser.add_argument('--use_x0_low_guidance', type=str2bool, default=False)
parser.add_argument('--use_x0_hres', type=str2bool, default=False)
parser.add_argument('--use_low_pass_x0_hres', type=str2bool, default=False)
parser.add_argument('--weight_x0_hres', metavar='N', type=float, nargs='+',
                    help='an integer for the accumulator')

parser.add_argument('--use_v_res', type=str2bool, default=False)
parser.add_argument('--use_low_pass_v_res', type=str2bool, default=False)
parser.add_argument('--weight_vres', metavar='N', type=float, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
iter_id = args.iter_id[0]
# print(args.use_x0_hres)
# print(args.use_swin_attention)
# print(args.use_low_pass_x0_hres)
# print(args.weight_x0_hres)

# print(args.use_v_res)
# print(args.use_low_pass_v_res)
# print(args.weight_vres)
# exit()
def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

seed = 3407
device = "cuda"
model_path = "black-forest-labs/FLUX.1-dev"
model_path = "/data_3/lijunjie/cache/huggingface/hub/FLUX.1-dev"
prompts_file = '/data_2/lijunjie/code/FreeScale-main/athenic_4K_prompt.txt'
# prompts_file = '/data_2/lijunjie/code/Hiresolution_generate/training_free/HiFlow-main/HiFlow-main/render.txt'
# prompts_file = '/data_2/lijunjie/code/Hiresolution_generate/training_free/HiFlow-main/HiFlow-main/render_ACM_MM.txt'
prompts = load_prompts(prompts_file)
save_folder = './save_result'
high_filter_ratio = 0.32
# print(args.use_x0_hres)
# print(args.use_swin_attention)
# print(args.use_low_pass_x0_hres)
# exit()
# print(args.weight_x0_hres)

# print(args.use_v_res)
# print(args.use_low_pass_v_res)
# print(args.weight_vres)
# exit()
##使用window attention
use_tvsda_attention = args.use_tvsda_attention
use_npa_attention = args.use_npa_attention
use_local_attention = args.use_local_attention
if use_tvsda_attention:
    method_name = 'tvsda_attention'
else:
    method_name = ''
if use_npa_attention:
    method_name = (method_name + '_' if method_name else '') + 'npa_attention'
if use_local_attention:
    method_name = (method_name + '_' if method_name else '') + 'local_attention'

##CSG
use_x0_hres = args.use_x0_hres
use_x0_low_guidance = args.use_x0_low_guidance
if not use_x0_low_guidance:
    method_name = method_name+'_'+'nox0_low_guidance'

if use_x0_hres:
    use_low_pass_x0_hres = args.use_low_pass_x0_hres
    weight_x0_hres = args.weight_x0_hres[0]
    if use_low_pass_x0_hres:
        method_name = method_name+'_add_-'+str(weight_x0_hres)+'hfreslargelowpass'+'_'+str(high_filter_ratio)
    else:
        method_name = method_name+'_add_-'+str(weight_x0_hres)+'hfres'
else:
    method_name = method_name

##CTG
use_v_res = args.use_v_res
if use_v_res:
    use_low_pass_v_res = args.use_low_pass_v_res
    weight_vres = args.weight_vres[0]
    method_name = method_name+'_add_-'+str(weight_vres)+'CTG'
    if use_low_pass_v_res:
        method_name = method_name+'lowpass'
else:
    method_name = method_name
# print(method_name)
# exit()
method_name = method_name
control_params={}
control_params['use_tvsda_attention'] = use_tvsda_attention
control_params['use_npa_attention'] = use_npa_attention
control_params['use_local_attention'] = use_local_attention
control_params['use_x0_hres'] = use_x0_hres
control_params['use_x0_low_guidance'] = use_x0_low_guidance
if use_x0_hres:
    control_params['use_low_pass_x0_hres'] = use_low_pass_x0_hres
    control_params['weight_x0_hres'] = weight_x0_hres

control_params['use_v_res'] = use_v_res
if use_v_res:
    control_params['use_low_pass_v_res'] = use_low_pass_v_res
    control_params['weight_vres'] = weight_vres

# method_name = 'hifill_4096'
ntk_factor = 10
target_heights = [2048,4096]
target_widths = [2048,4096]
method_name = str(target_heights[-1])+'_'+str(target_widths[-1])+'_'+method_name
text_duplication = False
if text_duplication:
    if len(target_heights)==0:
        method_name = method_name+'no_initial_guidance_ntk_factor_'+str(ntk_factor)+'usetext_duplication'
    else:
        method_name = method_name+'_ntk_factor_'+str(ntk_factor)+'usetext_duplication'
else:
    if len(target_heights)==0:
        method_name = method_name+'no_initial_guidance_ntk_factor_'+str(ntk_factor)
    else:
        method_name = method_name+'ntk_factor_'+str(ntk_factor)
use_loral = False
if use_loral:
    # method_name = method_name+'_aidma'
    method_name = method_name+'_wukong'

# method_name = method_name+'_test_new'
filter_ratio = 0.2
# print(method_name)
# exit()
method_name = method_name+'_test_debug'
# method_name = method_name+'_ourvis'
# folder_name = './save_result/_nox0_low_guidancedirect_inference'
# folder_name = save_folder+method_name+'_vis_intermediate_20'
# print(folder_name)
folder_name = save_folder
# exit()
os.makedirs(folder_name, exist_ok=True)
iter_num = int(len(prompts)/8)
# print(control_params)
# exit()
# print(iter_num)
# exit()
if True:
    transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16)
    pipe = FluxPipeline.from_pretrained(model_path, transformer=None,  torch_dtype=torch.float16)
    pipe.transformer = transformer
    pipe.scheduler.use_dynamic_shifting = False
    #如果不使用pg策略时需要打开下面的超参数。
    # pipe.scheduler.config.use_dynamic_shifting = False
    pipe.to(device)
# print('lllllllllllllll')
# exit()
# generate_image = "/data_3/lijunjie/Hiresoution_generate/HiFlow/save_result/2048_4096_swin_attention_add_-1.0hfreslargelowpass_0.32_add_-1.0crosstimex0lowpassntk_factor_10/0/"
# generate_files = glob.glob(generate_image+'*')
# generated_id = []
# for generate_file in generate_files:
#     file_id = generate_file.split('/')[-1][:-4]
#     generated_id.append(file_id)


# print(len(generate_file))
# exit()
# LoRA can be downloaded from https://civitai.com/models/832683/flux-pro-11-style-lora-extreme-detailer-for-flux-illustrious
if use_loral:
    # pipe.load_lora_weights("/data_3/lijunjie/pretrainmodel/aidmaFLUX/aidmaFLUXPro1.1-FLUX-v0.3.safetensors") # optional
    pipe.load_lora_weights("/data_3/lijunjie/pretrainmodel/black_myth/BLACK MYTH WUKONG.safetensors")
# vis_files = ['40398007404014594']
vis_files = []
start_time = time.time()
for index, prompt in enumerate(prompts[iter_id*iter_num:(iter_id+1)*iter_num]):
    set_seeds(seed)
    # print('index: ',index)
    # exit()
    if index>128:
        break
    

    # prompt = "A robot standing in the rain reading newspaper, rusty and worn down, in a dystopian cyberpunk street, photo-realistic, urbanpunk. aidmaFLUXPro1.1"
    file_id = prompt.split('txt')[0][:-1]
    # if not (file_id in generated_id):
    #     print(file_id)
    # continue
    prompt = prompt.split('txt')[1]
    # prompt ='A modern advertising poster with large text "FUTURE AI", minimalist design, high quality typography.'
    valid = True
    for vis_file in vis_files:
        if vis_file in file_id:
            valid = True
            break
    print(valid,file_id)
    if not valid:
        continue

    images,low_high_result = pipe(
        prompt = prompt,
        # --------- Default Inference Parameters for Flux-dev 1K generation -----------
        height = 1024,
        width = 1024,
        guidance_scale = 3.5,
        num_inference_steps = 30,
        max_sequence_length = 512,
        # -------- Flux High Resolution Inference Toolkits ----
        ntk_factor = [ntk_factor, ntk_factor], 
        proportional_attention = True, 
        text_duplication = text_duplication, 
        swin_pachify = False, 
        # --------------- HiFlow Parameters ---------
        target_heights = target_heights, 
        target_widths = target_widths, 
        num_inference_steps_highres = [16, 10,], 
        filter_ratio = [filter_ratio, filter_ratio,], 
        high_filter_ratio = high_filter_ratio,
        guidance_scale_highres = [4.5, 6,], 
        structure_guidance = "fft", # ["fft", "dwt"]
        alphas = [1.0, 0.25,], # structure guidance scale
        betas = [1.0, 0.25,], # acceleration guidance scale
        upsampling_choice = "latent", # ["latent", "pixel"]
        flow_choice = "v_theta",
        generator=torch.Generator("cuda").manual_seed(seed),
        control_params=control_params,
        )
    # break
    for i, result in enumerate(images):
        image = result.images[0]
        os.makedirs(folder_name+'/'+method_name+'/'+str(i), exist_ok=True)
        # if i == 2:
        #     new_size = (8192, 8192)  # 目标尺寸
        #     image = image.resize(new_size)
        #     print('ljj')
        image.save("{}/{}/{}/{}.jpg".format(folder_name, method_name, str(i),str(file_id)))
    # for scale in low_high_result.keys():
    #     if '2048_2048'==scale:
    #         scale_level = 1
    #     elif '4096_4096'==scale:
    #         scale_level = 2
    #     for t in low_high_result[scale].keys(): 
    #         name = file_id+'_'+str(t)
    #         low_high_result[scale][t][0].save("{}/{}/{}/{}.jpg".format(folder_name, method_name, str(scale_level),str(name)))
    # # exit()
    # images[0].save("hiflow.jpg")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds",elapsed_time/(60))