//index.js
//获取应用实例
const app = getApp()
var WxAutoImage = require('../../utils/WxAutoImage.js');

Page({
  data: {
    cGenerator: app.globalData.cGenerator,
    showLoading:false
  },
  //事件处理函数
  bindViewTap: function() {
    wx.navigateTo({
      url: '../logs/logs'
    })
  },
  onLoad: function () {
    var that = this;
    that.setData({
        motto: '今天点评是',
        buttonText: "懒得写评论？猛击我"
      })
  },
  cusImageLoad: function (e) {
    var that = this;
    //这里看你在wxml中绑定的数据格式 单独取出自己绑定即可
    that.setData(WxAutoImage.wxAutoImageCal(e));
  },


  cPredict: function (e) {
    var that = this;
    if (!app.globalData.cGenerator) {
      console.log(e),
      that.setData({
        motto: 'AI借鉴过众多美食点评后，生产专属您的评论',
        buttonText: '不适合？再折磨AI一下',
        showLoading:true
      }),
      app.globalData.cGenerator = true
    }
    else {
      that.setData({
        buttonText:'不适合？再折磨AI一下',
        motto: 'AI借鉴过众多美食点评后，生产专属您的评论',
        showLoading:true
      })
    }
    wx.request(
      {
        url: 'http://101.132.160.113:45723/predict',
        success: function (res) {
          that.setData({
            comments: res.data,
            showLoading:false
          })
        }
      })
  }
})