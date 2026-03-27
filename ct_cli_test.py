import yaml
import os
import subprocess
import typer
import pipe
from threading import Thread
import time
Config = yaml.load(open('config.yaml'),Loader=yaml.FullLoader)
app = typer.Typer()


@app.command()
def config_test():
    config = Config.get('ZolixMcController', None)
    if not config:
        typer.echo("转台控制器配置出错!请检查config.yaml文件")
        return
    typer.echo(type(config['speed']))
    typer.echo(type(config['degree']))
    typer.echo(360/config['degree'])



@app.command()
def snap():
    server_thread = Thread(target=pipe.detector_server, args=(r'\\.\pipe\detectResult', b'ctRestruct'), daemon=True)
    server_thread.start()
    py34 = os.environ.get('py34', None)
    if not py34:
        typer.echo("py34环境变量未配置!请检查系统环境变量, 保证py34环境变量指向3.4版本python.exe的路径")
        return
    sub = subprocess.Popen([py34, 'pipe_client_test.py'], stdin=subprocess.PIPE)
    for i in range(10):
        sub.stdin.write(b'snap\n')
        sub.stdin.flush()

        time.sleep(1)
    exit


if __name__ == "__main__":
    app()

