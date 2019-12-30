<template>
  <div id="app" style="height: 100%;">
    <el-container id="dashboard" style="height: 100%;">
      <el-header style="height: 60px; background-color: #598c88; color: #fff; padding: 0; line-height: 60px;">
        <h4 style="margin: 0px 0px 0px 1rem; display: inline;">OCR</h4>
        <el-dropdown style="float: right">
          <i class="el-icon-setting" style="margin-right: 15px"></i>
          <el-dropdown-menu slot="dropdown">
            <el-dropdown-item>用户管理</el-dropdown-item>
            <el-dropdown-item>用户注销</el-dropdown-item>
          </el-dropdown-menu>
        </el-dropdown>
        <span style="cursor: pointer; display: inline; float: right; margin-right: 1rem">admin</span>
      </el-header>
      <el-container>
        <el-aside style="width: initial; background-color: #545c64;">
          <el-menu default-active="1" @open="handleOpen" background-color="#545c64" text-color="#fff" @close="handleClose" active-text-color="#ffd04b">
            <el-menu-item index="1">
              <i class="el-icon-menu"></i>
              <span slot="title">票据识别</span>
            </el-menu-item>
            <el-menu-item index="2" disabled>
              <i class="el-icon-menu"></i>
              <span slot="title">身份证识别</span>
            </el-menu-item>
          </el-menu>
        </el-aside>
        <el-main>
          <el-container>
            <el-header>
              <el-row class="img-box">
                <el-upload
                  class="inline-block"
                  action="upload"
                  :http-request="El_upload"
                  :show-file-list="false"
                  :on-success="handleAvatarSuccess"
                  :on-change="handleChangeImg"
                  list-type="picture" accept="image/*">
                  <el-button type="primary" >选择图片</el-button>
                </el-upload>
                <el-button type="primary" @click="handleRunDetect">检测识别</el-button>
                <!--<input name="file" type="file" accept="image/*" @change="uploads"/>-->
              </el-row>
            </el-header>
            <el-main style="width: 88%; height: auto">
              <el-container >
                <!--<el-image :src="url" style="width:65%; height:60%;">加载图片</el-image>-->
                <div class="box">
                  <el-image class="box-img" v-if="imageUrl" fit="scale-down" :src="imageUrl" :onerror="handleDefaultPath" ></el-image>
                </div>
                <div class="box-mes">
                  <el-collapse v-model="activeNames" @change="handleChangeText">
                    <el-collapse-item title="发票基本信息" name="1">
                      <div>发票号码:{{this.collapseList.invoice_id}}</div>
                      <div>发票类型:{{this.collapseList.invoice_type}}</div>
                      <div>发票名称:{{this.collapseList.invoice_name}}</div>
                      <div>备注:{{this.collapseList.invoice_comment}}</div>
                    </el-collapse-item>
                    <el-collapse-item title="购买方" name="2">
                      <div>名称:{{this.collapseList.buy_name}}</div>
                      <div>纳税人识别号:{{this.collapseList.buy_id}}</div>
                      <div>地址、电话:{{this.collapseList.buy_address}}</div>
                      <div>开户行及账号:{{this.collapseList.buy_account}}</div>
                    </el-collapse-item>
                    <el-collapse-item title="销售方" name="3">
                      <div>名称:{{this.collapseList.sell_name}}</div>
                      <div>纳税人识别号:{{this.collapseList.sell_id}}</div>
                      <div>地址、电话:{{this.collapseList.sell_address}}</div>
                      <div>开户行及账号:{{this.collapseList.sell_account}}</div>
                    </el-collapse-item>
                    <el-collapse-item title="货物或应税劳务、服务名称" name="4">
                      <div>名称:{{}}</div>
                      <div>税率:{{}}</div>
                    </el-collapse-item>
                    <el-collapse-item title="合计" name="5">
                      <div>税价合计(小写):{{this.collapseList.total_money}}</div>
                    </el-collapse-item>
                  </el-collapse>
                </div>
              </el-container>
            </el-main>
            <el-footer></el-footer>
          </el-container>
        </el-main>
      </el-container>
    </el-container>

  </div>
</template>

<script>
  export default {
    name: 'App',
    data(){
      return {
        handleDefaultPath:"this.src='../static/template.png'",
        imageUrl:'../static/template.png',
        imageName:'',
        activeNames: ['1'],
        instance:null,  //axios 实例
        collapseList:[ //最终返回识别数据
          {
            invoice_id:'',
            invoice_name:'',
            invoice_type:'',
            invoice_comment:'',
            buy_name:'',
            buy_id:'',
            buy_address:'',
            buy_account:'',
            sell_name:'',
            sell_id:'',
            sell_address:'',
            sell_account:'',
            goods_tax:'',
            total_money:''
          }
        ]
      }
    },
    // 实例化一个新的axios
    created(){
      this.instance = this.$axios.create({
        baseURL:'/ocr'
      })
    },
    methods:{
      // 侧面菜单栏控制
      handleOpen(key, keyPath) {
        console.log(key, keyPath);
      },
      handleClose(key, keyPath) {
        console.log(key, keyPath);
      },
      // 文件上传
      El_upload(content){
        let form = new FormData();
        form.append('file', content.file);
        this.instance.post(content.action, form).then(res => {
          if (res.data.code != 0) {
            content.onError('文件上传失败, code:' + res.data.code)
          } else {
            this.url=content.file;
            content.onsuccess('文件上传成功！')
          }
        }).catch(error => {
          if (error.response) {
            content.onError('文件上传失败,' + error.response.data);
          } else if (error.request) {
            content.onError('文件上传失败，服务器端无响应')
          } else {
            content.onError('文件上传失败，请求封装失败')
          }
        });
      },
      // 上传成功时将图片添加到缓存区以方便获取
      handleAvatarSuccess(res, file){
        this.imageUrl = window.URL.createObjectURL(file.raw);
      },
      // 重定向加载图片
      handleChangeImg(file, fileList){
        let fileName = file.name;
        this.imageName = fileName;
        let regex = /(.jpg|.jpeg|.gif|.png|.bmp)$/;
        if (regex.test(fileName.toLowerCase())){
          this.imageUrl = file.url;
        } else {
          this.$message.error('请选择图片文件');
        }
      },
      // 下拉框的控制
      handleChangeText(val){
        console.log(val);
      },
      handleRunDetect(){
        this.instance.post('detect', {image_name:this.imageName}).then(res => {
          console.log(res)
        }).catch(res => {
          console.log(res)
        })
      },
      // 另一种上传文件方法,暂时没用
      uploads(e){
        let file = e.target.files[0];
        let param = new FormData();  // 创建form对象
        param.append('file', file, file.name);  // 通过append向form对象添加数据
        console.log(param.get('file')); // FormData私有类对象，访问不到，可以通过get判断值是否传进去
        let config = {
          headers: {'Content-Type': 'multipart/form-data'}
        };
        // 添加请求头
        this.instance.post('/upload', param, config).then(response => {
          if (response.data.code === 0) {self.ImgUrl = response.data.data}
          console.log(response.data)
        })
      }
    }
  }

</script>

<style>
  html {height: 100%;}
  body {
    height: 100%;
  }
  #app {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    cursor: pointer;
    display: inline;
    margin-right: 1rem;
  }
  .inline-block {
    display: inline-block;
  }
  .box {
    position: absolute;
    overflow: hidden;
    float: left;
    width: 1040px;
    height: 650px;
    vertical-align: middle;
    text-align: center;
    border:1px solid #000;
    background-color: #000;
  }
  .box-img {
    position: relative;
    width: 100%;
    height: 100%;
  }
  .box-mes {
    position: relative;
    margin-left: 1043px;
    height: 650px;
    width: 340px;
    background-color: #fafafa;
    word-wrap: break-word;
    word-break: break-all;
  }
</style>
