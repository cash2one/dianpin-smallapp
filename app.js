//app.js
App({
  onLaunch: function () {
    // 展示本地存储能力
    var logs = wx.getStorageSync('logs') || []
    logs.unshift(Date.now())
    wx.setStorageSync('logs', logs)
    
  },
  globalData: {
    cGenerator:false,
    comments:'',
    buttonText: '生成今日评论',
    motto:''
  }
})