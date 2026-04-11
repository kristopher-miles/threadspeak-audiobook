const basePackages = [
  "uv pip uninstall google-genai",
  "uv pip install -r requirements.txt",
]

const verifyTestEnv = [
  "python -c \"import fastapi, openai, pytest, uvicorn, pydantic, docx; print('Dependency check OK')\"",
]

module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "uv cache clean"
    }
  }, {
    when: "{{!which('sox')}}",
    method: "shell.run",
    params: {
      message: "conda install -y -c conda-forge sox"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "python -m venv env"
    }
  }, {
    when: "{{platform === 'darwin' && arch === 'arm64' && exists('models')}}",
    method: "fs.rm",
    params: {
      path: "models"
    }
  }, {
    when: "{{platform === 'darwin' && arch === 'arm64' && exists('app/models')}}",
    method: "fs.rm",
    params: {
      path: "app/models"
    }
  }, {
    when: "{{platform === 'darwin' && arch === 'arm64'}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        ...basePackages,
        "uv pip uninstall qwen-tts",
        "uv pip install --prerelease=allow mlx==0.30.3 mlx-lm==0.30.5 mlx-metal==0.30.3",
        "uv pip install git+https://github.com/Blaizzy/mlx-audio.git@9349644ccbd62eb10900852228f7b952c566def3",
        "uv pip install sentencepiece tiktoken",
        ...verifyTestEnv,
      ]
    }
  }, {
    when: "{{!(platform === 'darwin' && arch === 'arm64')}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        ...basePackages,
        "uv pip install qwen-tts==0.1.1",
        ...verifyTestEnv,
      ]
    }
  }, {
    when: "{{!(platform === 'darwin' && arch === 'arm64')}}",
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        path: "app",
        venv: "env",
        flashattention: true
      }
    }
  }, {
    method: "notify",
    params: {
      html: "Installation Complete! Click 'Start' to launch the application."
    }
  }]
}
