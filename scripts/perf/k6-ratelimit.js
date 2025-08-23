/**
 * k6 script to exercise rate limiting on a protected endpoint.
 *
 * Usage:
 *  BASE_URL=http://localhost:8000 k6 run scripts/perf/k6-ratelimit.js
 *  BASE_URL=http://localhost:8000 TOKEN="$(python scripts/smoke/jwt/gen_test_jwt.py --private-key ./private.pem --sub tester)" k6 run scripts/perf/k6-ratelimit.js
 *
 * Notes:
 * - Health endpoints are limiter-exempt per [python.function health()](fba_bench_api/main.py:226) and [python.function health_v1()](fba_bench_api/main.py:265); this script hits /api/v1/settings instead.
 * - Tune API_RATE_LIMIT in your environment to control the limit.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const TOKEN = __ENV.TOKEN || '';

export const options = {
  scenarios: {
    burst_then_hold: {
      executor: 'ramping-arrival-rate',
      startRate: 5,
      timeUnit: '1s',
      preAllocatedVUs: 20,
      maxVUs: 200,
      stages: [
        { target: 50, duration: '15s' },   // ramp up
        { target: 100, duration: '30s' },  // higher load to trigger 429s
        { target: 0, duration: '10s' },    // ramp down
      ],
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.10'], // up to 10% errors acceptable while limiting is validated
  },
};

function headers() {
  const h = { 'Content-Type': 'application/json' };
  if (TOKEN) {
    h['Authorization'] = `Bearer ${TOKEN}`;
  }
  return h;
}

export default function () {
  const res = http.get(`${BASE_URL}/api/v1/settings`, { headers: headers() });
  // Accept either 2xx or 429 for limiter behavior, flag others
  check(res, {
    'status is 2xx or 429': (r) => (r.status >= 200 && r.status < 300) || r.status === 429,
  });
  sleep(0.1);
}

export function handleSummary(data) {
  const counts = {
    '2xx': 0,
    '429': 0,
    other: 0,
  };
  const statusTrend = data.metrics.http_reqs_statuses || {};
  // iterate responses if available; fallback to simple log
  console.log('k6 summary:');
  console.log(` - Requests: ${data.metrics.http_reqs ? data.metrics.http_reqs.values.count : 'n/a'}`);
  // Best-effort log; k6 JS summary APIs don't expose per-status counts by default without custom metrics.
  console.log(' - Expect a mix of 2xx and 429 when hitting the limiter.');

  return {
    stdout: JSON.stringify(
      {
        info: 'Rate limit smoke summary',
        total_reqs: data.metrics.http_reqs ? data.metrics.http_reqs.values.count : null,
        http_req_failed: data.metrics.http_req_failed ? data.metrics.http_req_failed.values.rate : null,
        note: 'Look for 429s as load increases; health endpoints are exempt and not used here.',
      },
      null,
      2
    ),
  };
}