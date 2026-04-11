/**
 * Metadron Capital — PM2 Ecosystem Configuration (CORRECTED)
 * Unified process management for the entire platform.
 *
 * Changes from original:
 *   1. express-frontend production mode: "node dist/index.cjs" instead of "npm run dev"
 *   2. airllm-model-server: path confirmed (engine/bridges/airllm_model_server.py exists)
 *   3. ainewton-service: path fixed (was missing, now points to engine/bridges/ainewton_service.py wrapper)
 *   4. metadron-cube: path fixed (was missing, now points to engine/bridges/metadron_cube_service.py wrapper)
 *   5. news-engine: path confirmed with correct cwd
 *
 * Usage:
 *   pm2 start ecosystem.config.cjs                  # Start all services
 *   pm2 start ecosystem.config.cjs --only engine-api # Start single service
 *   pm2 start ecosystem.config.cjs --env production  # Production mode
 *   pm2 stop all                                     # Stop everything
 *   pm2 restart all                                  # Restart everything
 *   pm2 monit                                        # Live monitoring
 *   pm2 logs                                         # Aggregate logs
 */

const path = require('path');
const ROOT = __dirname;

module.exports = {
  apps: [
    // CORE PLATFORM SERVICES
    {
      name: 'engine-api',
      script: 'python3',
      args: '-m uvicorn engine.api.server:app --host 0.0.0.0 --port 8001 --log-level info',
      cwd: ROOT,
      interpreter: 'none',
      env: { ENGINE_API_PORT: '8001', PYTHONUNBUFFERED: '1' },
      env_production: { ENGINE_API_PORT: '8001', PYTHONUNBUFFERED: '1', NODE_ENV: 'production' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '2G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: path.join(ROOT, 'logs/pm2/engine-api-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/engine-api-out.log'),
      merge_logs: true, restart_delay: 5000, max_restarts: 10, min_uptime: '10s',
    },
    {
      name: 'express-frontend',
      script: process.env.NODE_ENV === 'production' ? 'node' : 'npm',
      args: process.env.NODE_ENV === 'production' ? 'dist/index.cjs' : 'run dev',
      cwd: ROOT, interpreter: 'none',
      env: { PORT: '5000', NODE_ENV: 'development' },
      env_production: { PORT: '5000', NODE_ENV: 'production' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '1G',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: path.join(ROOT, 'logs/pm2/express-frontend-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/express-frontend-out.log'),
      merge_logs: true, restart_delay: 3000, max_restarts: 10, min_uptime: '5s',
    },
    // MIROFISH
    {
      name: 'mirofish-backend', script: 'python3', args: 'run.py',
      cwd: path.join(ROOT, 'mirofish/backend'), interpreter: 'none',
      env: { FLASK_PORT: '5001', PYTHONUNBUFFERED: '1' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '1G',
      error_file: path.join(ROOT, 'logs/pm2/mirofish-backend-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/mirofish-backend-out.log'),
      merge_logs: true, restart_delay: 3000, max_restarts: 10,
    },
    {
      name: 'mirofish-frontend', script: 'npm', args: 'run dev',
      cwd: path.join(ROOT, 'mirofish/frontend'), interpreter: 'none',
      env: { PORT: '5174', NODE_ENV: 'development' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '512M',
      error_file: path.join(ROOT, 'logs/pm2/mirofish-frontend-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/mirofish-frontend-out.log'),
      merge_logs: true, restart_delay: 3000, max_restarts: 10,
    },
    // QWEN 2.5-7B MODEL SERVER
    {
      name: 'qwen-model-server', script: 'python3',
      args: ['web_demo.py', '--checkpoint-path', process.env.QWEN_MODEL_PATH || 'Qwen/Qwen2.5-Omni-7B', '--server-port', '7860', '--server-name', '0.0.0.0'].join(' '),
      cwd: path.join(ROOT, 'Qwen 2.5-7b'), interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', CUDA_VISIBLE_DEVICES: process.env.QWEN_GPU_DEVICES || '0', QWEN_SERVER_PORT: '7860' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '16G',
      error_file: path.join(ROOT, 'logs/pm2/qwen-model-server-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/qwen-model-server-out.log'),
      merge_logs: true, restart_delay: 10000, max_restarts: 5, min_uptime: '30s', kill_timeout: 30000,
    },
    // NEWS ENGINE
    {
      name: 'news-engine', script: 'index.js',
      cwd: path.join(ROOT, 'News engine'), interpreter: 'node',
      env: { NODE_ENV: 'development' }, env_production: { NODE_ENV: 'production' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '512M',
      error_file: path.join(ROOT, 'logs/pm2/news-engine-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/news-engine-out.log'),
      merge_logs: true, restart_delay: 5000, max_restarts: 10,
    },
    // LIVE TRADING LOOP
    {
      name: 'live-loop', script: 'python3',
      args: '-c "from engine.live_loop_orchestrator import LiveLoopOrchestrator; o = LiveLoopOrchestrator(); o.run()"',
      cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', METADRON_MODE: 'live' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '4G',
      error_file: path.join(ROOT, 'logs/pm2/live-loop-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/live-loop-out.log'),
      merge_logs: true, restart_delay: 10000, max_restarts: 5, min_uptime: '30s', kill_timeout: 15000,
    },
    // SCHEDULED TASKS
    {
      name: 'market-open', script: 'python3', args: 'run_open.py', cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1' }, cron_restart: '30 9 * * 1-5', autorestart: false, watch: false,
      error_file: path.join(ROOT, 'logs/pm2/market-open-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/market-open-out.log'), merge_logs: true,
    },
    {
      name: 'market-close', script: 'python3', args: 'run_close.py', cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1' }, cron_restart: '0 16 * * 1-5', autorestart: false, watch: false,
      error_file: path.join(ROOT, 'logs/pm2/market-close-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/market-close-out.log'), merge_logs: true,
    },
    {
      name: 'hourly-tasks', script: 'python3', args: 'run_hourly.py', cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1' }, cron_restart: '0 * * * *', autorestart: false, watch: false,
      error_file: path.join(ROOT, 'logs/pm2/hourly-tasks-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/hourly-tasks-out.log'), merge_logs: true,
    },
    // PLATFORM ORCHESTRATOR
    {
      name: 'platform-orchestrator', script: 'python3', args: 'platform_orchestrator.py',
      cwd: ROOT, interpreter: 'none', env: { PYTHONUNBUFFERED: '1' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '2G',
      error_file: path.join(ROOT, 'logs/pm2/platform-orchestrator-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/platform-orchestrator-out.log'),
      merge_logs: true, restart_delay: 5000, max_restarts: 10, min_uptime: '10s',
    },
    // LLM INFERENCE BRIDGE (Qwen + Air-LLM + Claude)
    {
      name: 'llm-inference-bridge', script: 'python3',
      args: '-m uvicorn engine.bridges.llm_inference_bridge:create_app --factory --host 0.0.0.0 --port 8002 --log-level info',
      cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', LLM_BRIDGE_PORT: '8002',
        QWEN_MODEL_PATH: process.env.QWEN_MODEL_PATH || 'Qwen/Qwen2.5-Omni-7B',
        AIRLLM_MODEL_PATH: process.env.AIRLLM_MODEL_PATH || 'meta-llama/Llama-3.1-70B',
        XIAOMI_MIMO_API_KEY: process.env.XIAOMI_MIMO_API_KEY || '' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '8G',
      error_file: path.join(ROOT, 'logs/pm2/llm-bridge-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/llm-bridge-out.log'),
      merge_logs: true, restart_delay: 5000, max_restarts: 10, min_uptime: '10s', kill_timeout: 30000,
    },
    // AI-NEWTON DISCOVERY SERVICE (FIX: now points to wrapper that delegates to worker)
    {
      name: 'ainewton-service', script: 'python3', args: 'engine/bridges/ainewton_service.py',
      cwd: ROOT, interpreter: 'none', env: { PYTHONUNBUFFERED: '1' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '4G',
      error_file: path.join(ROOT, 'logs/pm2/ainewton-service-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/ainewton-service-out.log'),
      merge_logs: true, restart_delay: 10000, max_restarts: 5, min_uptime: '30s', kill_timeout: 60000,
    },
    // CONTINUOUS LEARNING LOOP
    {
      name: 'learning-loop', script: 'python3', args: 'engine/learning/continuous_learning_loop.py',
      cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', LLM_BRIDGE_URL: 'http://localhost:8002' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '4G',
      error_file: path.join(ROOT, 'logs/pm2/learning-loop-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/learning-loop-out.log'),
      merge_logs: true, restart_delay: 10000, max_restarts: 10, min_uptime: '30s',
    },
    // METADRON CUBE — 24/7 Regime Detection (FIX: now points to service wrapper)
    {
      name: 'metadron-cube', script: 'python3', args: 'engine/bridges/metadron_cube_service.py',
      cwd: ROOT, interpreter: 'none',
      env: { PYTHONUNBUFFERED: '1', METADRON_CUBE_MODE: 'continuous' },
      instances: 1, autorestart: true, watch: false, max_memory_restart: '2G',
      error_file: path.join(ROOT, 'logs/pm2/metadron-cube-error.log'),
      out_file: path.join(ROOT, 'logs/pm2/metadron-cube-out.log'),
      merge_logs: true, restart_delay: 5000, max_restarts: 10, min_uptime: '10s',
    },
  ],
};
