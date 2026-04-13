module.exports = {
  run: [{
    method: "script.stop",
    params: {
      uri: ["start.js"]
    }
  }, {
    method: "fs.rm",
    params: {
      path: "runtime/current/project"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "runtime/runs"
    }
  }]
}
