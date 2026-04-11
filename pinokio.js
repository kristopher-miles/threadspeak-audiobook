const fs = require('fs')
const path = require('path')

module.exports = {
  version: "5.1",
  title: "Threadspeak",
  description: "Import your book, process it via LLM, then convert it into an audiobook read by full cast.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    // Check running states
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      test: info.running("test.js"),
      reset: info.running("reset.js"),
      update: info.running("update.js")
    }

    // Check file existence states
    let installed = info.exists("app/env")

    // Handle running states first
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js"
      }]
    }

    if (running.start) {
      let local = info.local("start.js")
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: local.url,
        }, {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Starting",
          href: "start.js",
        }]
      }
    }

    if (running.test) {
      return [{
        default: true,
        icon: "fa-solid fa-flask-vial",
        text: "Running Tests",
        href: "test.js"
      }]
    }

    if (running.reset) {
      return [{
        default: true,
        icon: "fa-solid fa-rotate-left",
        text: "Resetting",
        href: "reset.js"
      }]
    }

    if (running.update) {
      return [{
        default: true,
        icon: "fa-solid fa-arrows-rotate",
        text: "Updating",
        href: "update.js"
      }]
    }

    // STATE: NOT_INSTALLED - auto-run install
    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js"
      }]
    }

    // STATE: INSTALLED
    return [{
      default: true,
      icon: "fa-solid fa-power-off",
      text: "Start",
      href: "start.js"
    }, {
      icon: "fa-solid fa-flask-vial",
      text: "Run Tests",
      href: "test.js"
    }, {
      icon: "fa-solid fa-folder-open",
      text: "Open Voicelines",
      href: "voicelines"
    }, {
      icon: "fa-solid fa-arrows-rotate",
      text: "Update",
      href: "update.js"
    }, {
      icon: "fa-solid fa-plug",
      text: "Reinstall",
      href: "install.js"
    }, {
      icon: "fa-solid fa-rotate-left",
      text: "Reset",
      href: "reset.js"
    }]
  }
}
