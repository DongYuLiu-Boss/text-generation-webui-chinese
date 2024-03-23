import importlib
import math
import re
import traceback
from functools import partial
from pathlib import Path

import gradio as gr
import psutil
import torch
from transformers import is_torch_xpu_available

from modules import loaders, shared, ui, utils
from modules.logging_colors import logger
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_instruction_template,
    save_model_settings,
    update_model_parameters
)
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user

    # Finding the default values for the GPU and CPU memories
    total_mem = []
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            total_mem.append(math.floor(torch.xpu.get_device_properties(i).total_memory / (1024 * 1024)))
    else:
        for i in range(torch.cuda.device_count()):
            total_mem.append(math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)))

    default_gpu_mem = []
    if shared.args.gpu_memory is not None and len(shared.args.gpu_memory) > 0:
        for i in shared.args.gpu_memory:
            if 'mib' in i.lower():
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
            else:
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)

    while len(default_gpu_mem) < len(total_mem):
        default_gpu_mem.append(0)

    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    if shared.args.cpu_memory is not None:
        default_cpu_mem = re.sub('[a-zA-Z ]', '', shared.args.cpu_memory)
    else:
        default_cpu_mem = 0

    with gr.Tab("模型", elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: shared.model_name, label='模型', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button', interactive=not mu)
                            shared.gradio['load_model'] = gr.Button("加载", visible=not shared.settings['autoload_model'], elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['unload_model'] = gr.Button("卸载", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['reload_model'] = gr.Button("重新加载", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['save_model_settings'] = gr.Button("保存设置", elem_classes='refresh-button', interactive=not mu)

                    with gr.Column():
                        with gr.Row():
                            shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button', interactive=not mu)
                            shared.gradio['lora_menu_apply'] = gr.Button(value='应用LoRAs', elem_classes='refresh-button', interactive=not mu)

        with gr.Row():
            with gr.Column():
                shared.gradio['loader'] = gr.Dropdown(label="模型加载器", choices=loaders.loaders_and_params.keys(), value=None)
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            with gr.Blocks():
                                for i in range(len(total_mem)):
                                    shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"GPU :{i}内存占用设置(0为不限制)", maximum=total_mem[i], value=default_gpu_mem[i])

                                shared.gradio['cpu_memory'] = gr.Slider(label="CPU内存占用设置(0为不限制)", maximum=total_cpu_mem, value=default_cpu_mem)

                            with gr.Blocks():
                                shared.gradio['transformers_info'] = gr.Markdown('load-in-4bit相关选项:')
                                shared.gradio['compute_dtype'] = gr.Dropdown(label="计算数据类型", choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype)
                                shared.gradio['quant_type'] = gr.Dropdown(label="量化类型", choices=["nf4", "fp4"], value=shared.args.quant_type)

                            shared.gradio['hqq_backend'] = gr.Dropdown(label="hqq_backend", choices=["PYTORCH", "PYTORCH_COMPILE", "ATEN"], value=shared.args.hqq_backend)
                            shared.gradio['n_gpu_layers'] = gr.Slider(label="n-gpu-layers(同时使用CUP和GPU)", minimum=0, maximum=256, value=shared.args.n_gpu_layers, info='必须设置为大于0，才能使用您的GPU')
                            shared.gradio['n_ctx'] = gr.Slider(minimum=0, maximum=shared.settings['truncation_length_max'], step=256, label="n_ctx(上下文长度)", value=shared.args.n_ctx, info='如果你在加载模型时遇到内存不足的问题，可以尝试降低此值')
                            shared.gradio['tensor_split'] = gr.Textbox(label='tensor_split(针对多GPU运行，设置占比)', info='将模型跨多个 GPU 拆分的比例列表。示例：18,17')
                            shared.gradio['n_batch'] = gr.Slider(label="n_batch(批处理大小)", minimum=1, maximum=2048, step=1, value=shared.args.n_batch)
                            shared.gradio['threads'] = gr.Slider(label="threads(线程数)", minimum=0, step=1, maximum=32, value=shared.args.threads)
                            shared.gradio['threads_batch'] = gr.Slider(label="threads_batch(批处理线程数)", minimum=0, step=1, maximum=32, value=shared.args.threads_batch)
                            shared.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                            shared.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")
                            shared.gradio['model_type'] = gr.Dropdown(label="model_type(模型类型)", choices=["None"], value=shared.args.model_type or "None")
                            shared.gradio['pre_layer'] = gr.Slider(label="pre_layer(发送到GPU的层数)", minimum=0, maximum=100, value=shared.args.pre_layer[0] if shared.args.pre_layer is not None else 0)
                            shared.gradio['gpu_split'] = gr.Textbox(label='gpu-split', info='每张GPU使用的显存（以GB为单位），用逗号分隔的列表。示例：20,7,7')
                            shared.gradio['max_seq_len'] = gr.Slider(label='max_seq_len', minimum=0, maximum=shared.settings['truncation_length_max'], step=256, info='上下文长度。如果你的模型加载过程中出现内存不足的情况，可以尝试降低这个值', value=shared.args.max_seq_len)
                            with gr.Blocks():
                                shared.gradio['alpha_value'] = gr.Slider(label='alpha_value(模型上下文长度)', minimum=1, maximum=8, step=0.05, info='位置嵌入的NTK RoPE缩放alpha因子。推荐值（NTKv1）：1.5倍上下文为1.75，2倍上下文为2.5。使用这个或使用compress_pos_emb，不要同时使用。', value=shared.args.alpha_value)
                                shared.gradio['rope_freq_base'] = gr.Slider(label='rope_freq_base(另一种模型上下文长度设置，一般默认)', minimum=0, maximum=1000000, step=1000, info='如果大于0，将替代alpha_value使用。这两个值的关系是rope_freq_base = 10000 * alpha_value^(64/63)。', value=shared.args.rope_freq_base)
                                shared.gradio['compress_pos_emb'] = gr.Slider(label='compress_pos_emb(最早的模型上下文长度设置，一般不用)', minimum=1, maximum=8, step=1, info='位置嵌入的压缩因子。应设置为（上下文长度）/（模型的原始上下文长度）。等于1/rope_freq_scale。', value=shared.args.compress_pos_emb)

                            shared.gradio['autogptq_info'] = gr.Markdown('对于从Llama派生的模型，ExLlamav2_HF比AutoGPTQ更推荐')
                            shared.gradio['quipsharp_info'] = gr.Markdown('目前必须手动安装QuIP#')

                        with gr.Column():
                            shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit（量化加载精度）", value=shared.args.load_in_8bit)
                            shared.gradio['load_in_4bit'] = gr.Checkbox(label="load-in-4bit（量化加载精度）", value=shared.args.load_in_4bit)
                            shared.gradio['use_double_quant'] = gr.Checkbox(label="use_double_quant（使用二次量化）", value=shared.args.use_double_quant)
                            shared.gradio['use_flash_attention_2'] = gr.Checkbox(label="use_flash_attention_2", value=shared.args.use_flash_attention_2, info='在加载模型时，将use_flash_attention_2设置为True。注意力层进行优化，提速模型，差别不显著。')
                            shared.gradio['auto_devices'] = gr.Checkbox(label="auto-devices(自动判断调整GPU内存占用值)", value=shared.args.auto_devices)
                            shared.gradio['tensorcores'] = gr.Checkbox(label="tensorcores", value=shared.args.tensorcores, info='仅限NVIDIA：使用带有张量核心支持的llama-cpp-python编译版本。这会增加RTX卡上的性能。')
                            shared.gradio['streaming_llm'] = gr.Checkbox(label="streaming_llm", value=shared.args.streaming_llm, info='(experimental) Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.')
                            shared.gradio['attention_sink_size'] = gr.Number(label="attention_sink_size", value=shared.args.attention_sink_size, precision=0, info='StreamingLLM: number of sink tokens. Only used if the trimmed prompt doesn\'t share a prefix with the old prompt.')
                            shared.gradio['cpu'] = gr.Checkbox(label="cpu(使用CPU加载模型)", value=shared.args.cpu, info='llama.cpp：使用未编译GPU加速的llama-cpp-python。转换器：使用CPU模式的PyTorch')
                            shared.gradio['row_split'] = gr.Checkbox(label="row_split", value=shared.args.row_split, info='Split the model by rows across GPUs. This may improve multi-gpu performance.')
                            shared.gradio['no_offload_kqv'] = gr.Checkbox(label="no_offload_kqv", value=shared.args.no_offload_kqv, info='不要将 K, Q, V 卸载到 GPU。这样可以节省 VRAM，但会降低性能。')
                            shared.gradio['no_mul_mat_q'] = gr.Checkbox(label="no_mul_mat_q", value=shared.args.no_mul_mat_q, info='禁用mulmat内核')
                            shared.gradio['triton'] = gr.Checkbox(label="triton", value=shared.args.triton)
                            shared.gradio['no_inject_fused_attention'] = gr.Checkbox(label="no_inject_fused_attention", value=shared.args.no_inject_fused_attention, info='Disable fused attention. Fused attention improves inference performance but uses more VRAM. Fuses layers for AutoAWQ. Disable if running low on VRAM.')
                            shared.gradio['no_inject_fused_mlp'] = gr.Checkbox(label="no_inject_fused_mlp", value=shared.args.no_inject_fused_mlp, info='Affects Triton only. Disable fused MLP. Fused MLP improves performance but uses more VRAM. Disable if running low on VRAM.')
                            shared.gradio['no_use_cuda_fp16'] = gr.Checkbox(label="no_use_cuda_fp16", value=shared.args.no_use_cuda_fp16, info='不使用fp16,这可以使某些系统上的模型运行得更快')
                            shared.gradio['desc_act'] = gr.Checkbox(label="desc_act", value=shared.args.desc_act, info='desc_act、wbits和groupsize用于没有quantize_config.json的旧模型')
                            shared.gradio['no_mmap'] = gr.Checkbox(label="no-mmap", value=shared.args.no_mmap)
                            shared.gradio['mlock'] = gr.Checkbox(label="mlock", value=shared.args.mlock)
                            shared.gradio['numa'] = gr.Checkbox(label="numa", value=shared.args.numa, info='NUMA支持可以帮助一些具有非均匀内存访问的系统')
                            shared.gradio['disk'] = gr.Checkbox(label="disk(使用磁盘加载模型)", value=shared.args.disk)
                            shared.gradio['bf16'] = gr.Checkbox(label="bf16(非量化加载精度)", value=shared.args.bf16)
                            shared.gradio['cache_8bit'] = gr.Checkbox(label="cache_8bit", value=shared.args.cache_8bit, info='使用8位缓存来节省显存。')
                            shared.gradio['cache_4bit'] = gr.Checkbox(label="cache_4bit", value=shared.args.cache_4bit, info='使用Q4缓存来节省显存。')
                            shared.gradio['autosplit'] = gr.Checkbox(label="autosplit", value=shared.args.autosplit, info='自动将模型张量分散到可用的GPU上')
                            shared.gradio['no_flash_attn'] = gr.Checkbox(label="no_flash_attn", value=shared.args.no_flash_attn, info='强制不使用flash-attention')
                            shared.gradio['cfg_cache'] = gr.Checkbox(label="cfg-cache", value=shared.args.cfg_cache, info='使用此加载器时需要使用CFG')
                            shared.gradio['num_experts_per_token'] = gr.Number(label="Number of experts per token", value=shared.args.num_experts_per_token, info='这只适用于像Mixtral这样的MoE（混合专家）模型')
                            with gr.Blocks():
                                shared.gradio['trust_remote_code'] = gr.Checkbox(label="trust-remote-code", value=shared.args.trust_remote_code, info='在加载分词器/模型时，将trust_remote_code设置为True。要启用此选项，请使用–trust-remote-code标志启动Web UI。', interactive=shared.args.trust_remote_code)
                                shared.gradio['no_use_fast'] = gr.Checkbox(label="no_use_fast(禁用快速加载器)", value=shared.args.no_use_fast, info='在加载分词器时，将use_fast设置为False。')
                                shared.gradio['logits_all'] = gr.Checkbox(label="logits_all", value=shared.args.logits_all, info='需要为此加载器设置，以便与 perplexity 评估一起工作。否则，忽略它，因为它会使提示处理变慢')

                            shared.gradio['disable_exllama'] = gr.Checkbox(label="disable_exllama(关闭exllama)", value=shared.args.disable_exllama, info='对于GPTQ模型，禁用ExLlama内核。')
                            shared.gradio['disable_exllamav2'] = gr.Checkbox(label="disable_exllamav2(关闭exllama2)", value=shared.args.disable_exllamav2, info='对于GPTQ模型，禁用ExLlamav2内核。')
                            shared.gradio['gptq_for_llama_info'] = gr.Markdown('Legacy loader for compatibility with older GPUs. ExLlamav2_HF or AutoGPTQ are preferred for GPTQ models when supported.')
                            shared.gradio['exllamav2_info'] = gr.Markdown("ExLlamav2_HF is recommended over ExLlamav2 for better integration with extensions and more consistent sampling behavior across loaders.")
                            shared.gradio['llamacpp_HF_info'] = gr.Markdown("llamacpp_HF将llama.cpp作为一个Transformers模型来加载。要使用它，你需要将你的GGUF放在models/子文件夹中，并放置必要的分词器文件。你可以使用“llamacpp_HF创建者”菜单来自动完成这个过程。")

            with gr.Column():
                with gr.Row():
                    shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='Autoload the model', info='是否在模型下拉菜单中选择模型时立即加载模型', interactive=not mu)

                with gr.Tab("下载"):
                    shared.gradio['custom_model_menu'] = gr.Textbox(label="下载模型和LoRA", info="输入Hugging Face用户名/模型路径，例如：facebook/galactica-125m。要指定分支，请在末尾添加一个“\”:”字符后跟分支名，如下所示：facebook/galactica-125m:main。要下载单个文件，请在第二个框中输入其名称", interactive=not mu)
                    shared.gradio['download_specific_file'] = gr.Textbox(placeholder="File name (for GGUF models)", show_label=False, max_lines=1, interactive=not mu)
                    with gr.Row():
                        shared.gradio['download_model_button'] = gr.Button("下载", variant='primary', interactive=not mu)
                        shared.gradio['get_file_list'] = gr.Button("获取文件列表", interactive=not mu)

                with gr.Tab("llamacpp_HF创建者"):
                    with gr.Row():
                        shared.gradio['gguf_menu'] = gr.Dropdown(choices=utils.get_available_ggufs(), value=lambda: shared.model_name, label='选择你的GGUF', elem_classes='slim-dropdown', interactive=not mu)
                        ui.create_refresh_button(shared.gradio['gguf_menu'], lambda: None, lambda: {'choices': utils.get_available_ggufs()}, 'refresh-button', interactive=not mu)

                    shared.gradio['unquantized_url'] = gr.Textbox(label="输入原始（未量化）模型的网址", info="Example: https://huggingface.co/lmsys/vicuna-13b-v1.5", max_lines=1)
                    shared.gradio['create_llamacpp_hf_button'] = gr.Button("提交", variant="primary", interactive=not mu)
                    gr.Markdown("这将把您的 gguf 文件移动到 models 文件夹的一个子文件夹中，同时还包括必要的分词器文件")

                with gr.Tab("自定义指令模板"):
                    with gr.Row():
                        shared.gradio['customized_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), value='None', label='选择所需的指令模板', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['customized_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)

                    shared.gradio['customized_template_submit'] = gr.Button("Submit", variant="primary", interactive=not mu)
                    gr.Markdown("这允许您为'模型加载器'菜单中当前选定的模型设置一个自定义模板。每当模型被加载时，将使用这个模板代替模型元数据中指定的模板，有时候元数据中的模板可能是错误的")

                with gr.Row():
                    shared.gradio['model_status'] = gr.Markdown('没有加载模型' if shared.model_name == 'None' else 'Ready')


def create_event_handlers():
    shared.gradio['loader'].change(
        loaders.make_loader_params_visible, gradio('loader'), gradio(loaders.get_all_params())).then(
        lambda value: gr.update(choices=loaders.get_model_types(value)), gradio('loader'), gradio('model_type'))

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        apply_model_settings_to_state, gradio('model_menu', 'interface_state'), gradio('interface_state')).then(
        ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
        update_model_parameters, gradio('interface_state'), None).then(
        load_model_wrapper, gradio('model_menu', 'loader', 'autoload_model'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['load_model'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['reload_model'].click(
        unload_model, None, None).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['unload_model'].click(
        unload_model, None, None).then(
        lambda: "Model unloaded", None, gradio('model_status'))

    shared.gradio['save_model_settings'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        save_model_settings, gradio('model_menu', 'interface_state'), gradio('model_status'), show_progress=False)

    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, gradio('lora_menu'), gradio('model_status'), show_progress=False)
    shared.gradio['download_model_button'].click(download_model_wrapper, gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['get_file_list'].click(partial(download_model_wrapper, return_links=True), gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['autoload_model'].change(lambda x: gr.update(visible=not x), gradio('autoload_model'), gradio('load_model'))
    shared.gradio['create_llamacpp_hf_button'].click(create_llamacpp_hf, gradio('gguf_menu', 'unquantized_url'), gradio('model_status'), show_progress=True)
    shared.gradio['customized_template_submit'].click(save_instruction_template, gradio('model_menu', 'customized_template'), gradio('model_status'), show_progress=True)


def load_model_wrapper(selected_model, loader, autoload=False):
    if not autoload:
        yield f"The settings for `{selected_model}` have been updated.\n\nClick on \"Load\" to load it."
        return

    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            yield f"Loading `{selected_model}`..."
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                output = f"Successfully loaded `{selected_model}`."

                settings = get_model_metadata(selected_model)
                if 'instruction_template' in settings:
                    output += '\n\nIt seems to be an instruction-following model with template "{}". In the chat tab, instruct or chat-instruct modes should be used.'.format(settings['instruction_template'])

                yield output
            else:
                yield f"Failed to load `{selected_model}`."
        except:
            exc = traceback.format_exc()
            logger.error('Failed to load the model.')
            print(exc)
            yield exc.replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Successfuly applied the LoRAs")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)

        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        if return_links:
            output = "```\n"
            for link in links:
                output += f"{Path(link).name}" + "\n"

            output += "```"
            yield output
            return

        yield ("Getting the output folder")
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp)
        if check:
            progress(0.5)

            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"Downloading file{'s' if len(links) > 1 else ''} to `{output_folder}`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)

            yield (f"Model successfully saved to `{output_folder}/`.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def create_llamacpp_hf(gguf_name, unquantized_url, progress=gr.Progress()):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(unquantized_url, None)

        yield ("Getting the tokenizer files links from Hugging Face")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=True)
        output_folder = Path(shared.args.model_dir) / (re.sub(r'(?i)\.gguf$', '', gguf_name) + "-HF")

        yield (f"Downloading tokenizer to `{output_folder}`")
        downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=False)

        # Move the GGUF
        (Path(shared.args.model_dir) / gguf_name).rename(output_folder / gguf_name)

        yield (f"Model saved to `{output_folder}/`.\n\nYou can now load it using llamacpp_HF.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
            return state['n_ctx']

    return current_length
