# coding: utf-8
# 用了 Python 的 Tornado 框架，当然也可以使用其它框架
# 英文文档 (v5.0) http://www.tornadoweb.org/en/stable/
# 中文文档 (v4.3) http://tornado-zh.readthedocs.io/zh/latest/
import json
import os

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
from tornado.escape import json_decode, json_encode
from tornado.concurrent import Future

# 这里导入模型实例
from RMA_recognition import RMA_model_instance

# 定义端口默认值
define("port", default=8080, help="run on the given port", type=int)

# 业务层，负责调用模型 API
class RecognitionService(object):
    def upload_image(self, file_metas):
        file_path = None
        if (file_metas):
            for meta in file_metas:
                # 上传图片保存路径为 {当前目录}/realimg
                upload_path = os.path.join(os.path.dirname(__file__), "realimg")
                
                filename = meta['filename']
                file_path = os.path.join(upload_path, filename)

                with open(file_path, 'wb') as f:
                    f.write(meta['body'])
       
        return file_path

    def recognition_model_run(self, image_path):
        res = dict(
            rtn = 0,
            msg = "",
            data = {}
        )
        # 调用模型 API
        try:
            data = RMA_model_instance.image_recognition(image_path)
            res = dict(
                rtn = 200,
                msg = "成功",
                data = data
            )
        except Exception as e:
            res["rtn"] = 500
            res["msg"] = str(e)
        
        return res


# 同步 Handler 示例
class SyncHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.recognition_service = RecognitionService()

    def post(self):
        file_metas = self.request.files.get("img")

        file_path = self.recognition_service.upload_image(file_metas)

        res = self.recognition_service.recognition_model_run(file_path)

        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json_encode(res))

        self.finish()
        



if __name__ == "__main__":
    tornado.options.parse_command_line()
    # 下面用正则表达式匹配 url
    app = tornado.web.Application(handlers=[
        (r"/api/recognition", SyncHandler)
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

# http://localhost:8080/api/async_api
# http://localhost:8080/api/sync_api
