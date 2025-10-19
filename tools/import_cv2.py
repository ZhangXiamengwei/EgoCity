import cv2
import os


def extract_frames(video_path, output_folder, interval=10, frames_per_interval=3):
    """
    从视频中按指定间隔提取帧。

    Args:
        video_path (str): 输入视频的路径
        output_folder (str): 输出帧图像的文件夹路径
        interval (int): 时间间隔（秒），默认为10秒
        frames_per_interval (int): 每个间隔提取的帧数，默认为3帧

    Returns:
        None
    """
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    
    # 获取视频的帧率（fps）
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # 获取视频的总帧数
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 获取视频的时长（秒）
    duration = total_frames / fps
    
    # 如果输出文件夹不存在，则创建文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 初始化当前帧计数器
    current_frame = 0
    
    # 用于记录抽取的帧数
    extracted_frame_count = 0
    
    # 遍历视频，抽取指定的帧
    while current_frame < total_frames:
        # 跳到视频的特定帧
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # 读取当前帧
        ret, frame = video.read()
        
        # 如果读取成功
        if ret:
            # 计算当前时间（秒）
            current_time = current_frame / fps
            
            # 如果当前时间是间隔的开始时间
            if current_time % interval < (interval / frames_per_interval):
                # 创建文件名
                output_filename = os.path.join(
                    output_folder, 
                    f"frame_{extracted_frame_count:04d}.jpg"
                )
                
                # 保存帧（添加压缩质量参数，95是一个比较好的质量值）
                save_success = cv2.imwrite(output_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # 检查是否保存成功
                if not save_success:
                    print(f"警告：帧 {extracted_frame_count} 保存失败")
                    continue
                
                # 增加已保存的帧数
                extracted_frame_count += 1
        
        # 每隔一定的时间抽取 3 帧
        current_frame += fps / frames_per_interval
    
    # 释放视频对象
    video.release()
    print(f"已成功提取 {extracted_frame_count} 帧图像到 {output_folder}")


if __name__ == '__main__':
    # 使用函数
    video_path = os.path.join("0224_test_video", "VID_0003.MOV")  # 替换为你的视频路径
    output_folder = "0224_test_picture"  # 输出帧的文件夹路径
    
    extract_frames(video_path, output_folder)
