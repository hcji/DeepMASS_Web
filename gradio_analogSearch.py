import gradio as gr
from backend.utils.theme import Seafoam
import pandas as pd
import threading
from analogSearch.gradio_service_search import (load_files, matchms_click_fn,
                                           show_information, clear_files,
                                           save_search_csv, init_custom_database,
                                           clear_custom_database)

from analogSearch.plot_utils import show_mol_search, show_ref_spectrum_search
from analogSearch.clean_files import monitor_directory
from backend.utils.auth import auth_ps


seafoam = Seafoam()


custom_css = """
footer {visibility:hidden;}
"""



# 界面构建
with gr.Blocks(title="Analog Search", css=custom_css, theme=seafoam) as demo:
    gr.Markdown("# 🔍 Analog Search")
    # 保存数据库路径
    # db_path_state = gr.State("data/all_data.mgf")
    # 保存读取文件的结果
    res_state = gr.State(pd.DataFrame())
    # 保存当前选择的structure
    structure_state = gr.State([])
    # 保存运行之后的result_df结果
    result_df = gr.State(pd.DataFrame())
    # 保存压缩文件目标名称
    target_zip_file_name_state = gr.State([])
    # save_search_csv 产生的临时目录
    temp_dir = gr.State("")

    # 用来存放动态生成的 reference list
    refs_pos_state = gr.State(None)
    refs_neg_state = gr.State(None)
    # 用来存放动态生成的 HNSW 索引对象
    hnsw_pos_state = gr.State(None)
    hnsw_neg_state = gr.State(None)
    # 保存 init_custom_database 创建的临时目录
    out_dir_state = gr.State("")     

    # 上传区 + 数据库选择
    with gr.Row():
        file_obj = gr.File(
            label="Unknown Compound",
            file_count="multiple",   # 允许多文件
            type="filepath", # 返回文件在服务器上的路径（字符串)
            file_types=[".mgf", ".msp", ".mat"],        
            height= 100
        )


        download = gr.File(visible=False, interactive=False)
    with gr.Accordion("Settings", open=False):
        with gr.Group():
            with gr.Row(equal_height=True):
                # 左侧：数据库选择
                db_option = gr.Radio(
                    ["Default", "Custom"],
                    label="Database",
                    value="Default",
                    # interactive=True,
                    container=True,
                    scale=1,
                )

                # 中间：文件上传（默认隐藏）
                db_upload = gr.File(
                    label="Custom Database",
                    file_types=[".mgf", ".msp", ".mat"],
                    visible=False,
                    scale=1
                )

                # 右侧：数值输入框
                threshold = gr.Number(
                    label="StructSimScore threshold",
                    value=0.85,          # 默认值
                    precision=2,        # 小数点后 2 位
                    interactive=True,
                    scale=1
                )

    # 按钮区
    with gr.Row():
        test_btn = gr.Button("Test File")

        run_btn = gr.Button("▶ Run Search")
        save_btn = gr.Button("Save Results")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Navigator")
            navigator_obj = gr.DataFrame(
                headers=["name"],
                elem_classes=["scroll"],
                value=pd.DataFrame(columns=["name"]),
                interactive=False,
                height=200,
                wrap=True,
            )


    with gr.Row():
        with gr.Tab(label="Spectrum"):
            with gr.Row():
                spectrum_plot_fig = gr.Plot(label="Spectrum")
                spectrum_loss_plot_fig = gr.Plot(label="Loss")

        with gr.Tab(label="Structure"):
            with gr.Row():
                ref_structure_fig = gr.Image(
                    label="Reference Structure", height=200, width=200, interactive=False
                )
        with gr.Tab(label="Information"):
            information_obj = gr.DataFrame(
                interactive=False,
                wrap=True,
                elem_classes=["scroll"],
                # height=200,
                value=pd.DataFrame(),
            )

    db_option.change(
        fn=lambda choice: (
            # 清 refs_pos_state..hnsw_neg_state四个 state
            None, None, None, None,
            # 控制上传框：Custom 时可见，否则隐藏并清空已选文件
            gr.update(visible=(choice == "Custom"), value=None),
            gr.update(interactive=(choice != "Custom"))
        ),
        inputs=[db_option],
        outputs=[
            refs_pos_state, refs_neg_state,
            hnsw_pos_state, hnsw_neg_state,
            db_upload, run_btn
        ]
    )

    db_upload.upload(
        fn=init_custom_database,
        inputs=[db_upload],
        outputs=[refs_pos_state, refs_neg_state,
                 hnsw_pos_state, hnsw_neg_state, 
                 out_dir_state, run_btn],
    )

    db_upload.clear(
        fn=clear_custom_database,
        inputs=[out_dir_state],
        outputs=[
            refs_pos_state, refs_neg_state,
            hnsw_pos_state, hnsw_neg_state,
            run_btn, db_upload
        ],
    )



    # 上传文件自动更新
    file_obj.upload(
        load_files,
        inputs=[file_obj],
        outputs=[res_state, navigator_obj, target_zip_file_name_state],
    )



    # Test 按钮：没有上传就走测试文件逻辑
    test_btn.click(
        fn=lambda: (
            # 1) 更新 file_obj 的值为测试文件路径列表
            gr.update(value=["analogSearch_data/test.mgf"]),
            # 2–4) 调用 load_files(None, paths) 拿到三个返回
            *load_files(["analogSearch_data/test.mgf"], None)
        ),
        inputs=[],  # 按钮不依赖任何输入
        outputs=[
            file_obj,                 # 对应 gr.update
            res_state,                # load_files 返回的 spectrums_df
            navigator_obj,            # load_files 返回的 name_list
            target_zip_file_name_state# load_files 返回的 zip 名
        ],
        api_name="load_test_file",
    )

    
    # 删除文件清除状态
    file_obj.clear(
        clear_files,
        inputs=[temp_dir],
        outputs=[
            res_state,
            structure_state,
            result_df,
            navigator_obj,
            information_obj,
            spectrum_plot_fig,
            spectrum_loss_plot_fig,
            ref_structure_fig,
            download,
            
        ],
    )


    # 点击按钮运行
    run_btn.click(
        fn=matchms_click_fn,
        inputs=[threshold, res_state,
                refs_pos_state, refs_neg_state,
                hnsw_pos_state, hnsw_neg_state],

        outputs=[res_state, result_df, information_obj,
                 spectrum_loss_plot_fig, spectrum_plot_fig, ref_structure_fig],
        concurrency_limit=4,
    )


    # 3) 点击 Navigator 行: 更新 Information
    navigator_obj.select(
        fn=show_information,
        inputs=[result_df],
        outputs=[information_obj],
     )
    # 选中 Navigator 行：更新质谱双图
    navigator_obj.select(
        fn=show_ref_spectrum_search,
        inputs=[res_state],
        outputs=[spectrum_loss_plot_fig, spectrum_plot_fig],
    )

    # 选中 Navigator 行：更新smiles分子图
    navigator_obj.select(
        fn=show_mol_search,
        inputs=[res_state],
        outputs=[ref_structure_fig],
    )

    # 点击保存按钮
    save_btn.click(
        fn=save_search_csv,
        inputs=[res_state, target_zip_file_name_state],
        outputs=[download, temp_dir],
    )

    # test_btn.click(
    #     fn=lambda: load_files(["data/test.mgf"], None),
    #     inputs=[],  # 不依赖其它组件
    #     outputs=[
    #         res_state,                # 更新结果状态
    #         navigator_obj,            # 更新导航列表
    #         target_zip_file_name_state # 更新 zip 名称
    #     ],
    # )

if __name__ == "__main__":

    # 启动 Gradio 服务之前，先启动清理线程
    tmp_dirs = ["tmp/database_tmp", "tmp/result_csv_tmp"]
    t = threading.Thread(
        target=monitor_directory,
        args=(tmp_dirs, 60*60*3, 3),  # 每 3 小时检查一次
        daemon=True              # 守护线程，主进程退出时自动结束
    )
    t.start()

    demo.launch(
        server_name="0.0.0.0",
        server_port=5578,
        # root_path="/anal_sear",
        show_api=False,
        auth=auth_ps,
        max_threads=2,
        auth_message="Please input your email and password",
    )

