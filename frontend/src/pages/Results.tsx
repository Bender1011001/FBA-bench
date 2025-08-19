import React, { useEffect, useState } from 'react';
import Skeleton from '../components/ui/Skeleton';
import Table, { type Column } from '../components/ui/Table';
import Badge from '../components/ui/Badge';
import { getEngineReport, getExperiments, RouteNotAvailableError } from '../services/api';
import type { EngineReport, Experiment } from '../types/api';
import { connectRealtime, EXPERIMENT_TOPIC_PREFIX } from '../services/realtime';
import useToast from '../contexts/useToast';

const Results: React.FC = () => {
  const toast = useToast();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [report, setReport] = useState<EngineReport | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingReport, setLoadingReport] = useState(false);

  const loadExperiments = async (signal?: AbortSignal) => {
    try {
      setLoadingList(true);
      const list = await getExperiments(signal);
      setExperiments(list);
      if (list.length && !selectedId) {
        setSelectedId(list[0].id);
      }
    } catch (e) {
      if (e instanceof RouteNotAvailableError) {
        toast.error('Experiments API not available (404).', 5000);
      } else {
        const msg = e instanceof Error ? e.message : 'Failed to load experiments.';
        toast.error(msg, 4000);
      }
    } finally {
      setLoadingList(false);
    }
  };

  const loadReport = async (id: string, signal?: AbortSignal) => {
    if (!id) return;
    try {
      setLoadingReport(true);
      const rep = await getEngineReport(id, signal);
      setReport(rep);
    } catch {
      // no report yet; keep UI usable
      setReport(null);
    } finally {
      setLoadingReport(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    loadExperiments(controller.signal);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    loadReport(selectedId, controller.signal);
    return () => controller.abort();
  }, [selectedId]);

  // realtime: refresh when experiment completes
  useEffect(() => {
    if (!selectedId) return;
    const client = connectRealtime();
    const topic = `${EXPERIMENT_TOPIC_PREFIX}${selectedId}`;
    client.subscribe(topic, (msg: unknown) => {
      const m = msg as { event?: string } | null;
      if (m?.event === 'experiment.completed') {
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        loadReport(selectedId);
        toast.success('Experiment completed. Results refreshed.', 4000);
      }
    });
    return () => {
      client.unsubscribe(topic);
      client.close();
    };
  }, [selectedId, toast]);

  const totals = report?.totals ?? {};
  const keyMetric = (name: string): number | string | null => {
    const keys = Object.keys(totals);
    const found = keys.find((k) => k.toLowerCase() === name.toLowerCase());
    return found ? (totals as Record<string, number | string>)[found] : null;
  };

  const scenarioColumns: Column<NonNullable<EngineReport>['scenario_reports'][number]>[] = [
    { key: 'scenario_key', header: 'Scenario' },
    {
      key: 'metrics',
      header: 'Metrics',
      render: (row) => {
        const m = row.metrics || {};
        const entries = Object.entries(m);
        if (!entries.length) return <span className="text-sm text-gray-600">—</span>;
        return (
          <div className="flex flex-wrap gap-2">
            {entries.map(([k, v]) => (
              <Badge key={k} tone="neutral" aria-label={`${k}: ${v}`}>
                {k}: {typeof v === 'number' ? v.toFixed(3) : String(v)}
              </Badge>
            ))}
          </div>
        );
      },
    },
    {
      key: 'validators',
      header: 'Validators',
      render: (row) => {
        const validators = row.validators || {};
        const entries = Object.entries(validators);
        if (!entries.length) return <span className="text-sm text-gray-600">—</span>;
        return (
          <div className="flex flex-wrap gap-1">
            {entries.map(([name, val]) => {
              const ok = typeof val === 'boolean' ? val : null;
              const tone: React.ComponentProps<typeof Badge>['tone'] = ok === true ? 'success' : ok === false ? 'danger' : 'warning';
              const symbol = ok === true ? '✓' : ok === false ? '✗' : 'i';
              return (
                <Badge key={name} tone={tone} aria-label={`${name}: ${String(val)}`}>
                  {name} {symbol}
                </Badge>
              );
            })}
          </div>
        );
      },
    },
  ];

  return (
    <section aria-labelledby="results-title" className="w-full h-full">
      <header className="mb-4">
        <h1 id="results-title" className="text-2xl font-semibold text-gray-900">
          Results
        </h1>
        <p className="text-gray-600">Analyze experiment outputs and KPIs.</p>
      </header>

      <div className="mb-4 flex flex-wrap gap-3 items-end">
        <div className="flex flex-col">
          <label htmlFor="exp-select" className="text-sm text-gray-700">
            Experiment
          </label>
          {loadingList ? (
            <div className="mt-1">
              <Skeleton variant="rect" width={260} height={36} />
            </div>
          ) : (
            <select
              id="exp-select"
              className="mt-1 px-2 py-2 border rounded text-sm min-w-[260px]"
              aria-label="Select experiment"
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
            >
              {experiments.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name} ({e.status})
                </option>
              ))}
              {!experiments.length && <option value="">No experiments</option>}
            </select>
          )}
        </div>
        <button
          type="button"
          onClick={() => selectedId && loadReport(selectedId)}
          className="px-3 py-2 rounded-md border text-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
          aria-label="Refresh results"
        >
          Refresh
        </button>
      </div>

      {/* Key metrics */}
      <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        {loadingReport ? (
          <>
            <Skeleton variant="card" />
            <Skeleton variant="card" />
            <Skeleton variant="card" />
            <Skeleton variant="card" />
          </>
        ) : (
          <>
            {report && Object.keys(totals).length > 0 ? (
              <>
                {['Accuracy', 'Technical Performance', 'Cost Efficiency'].map((label) => {
                  const v = keyMetric(label);
                  if (v === null) return null;
                  return (
                    <div key={label} className="rounded-lg border bg-white p-4">
                      <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
                      <div className="mt-1 text-2xl font-semibold text-gray-900">
                        {typeof v === 'number' ? v.toFixed(3) : String(v)}
                      </div>
                    </div>
                  );
                })}
                {/* Render first few additional totals */}
                {Object.entries(totals)
                  .filter(([k]) => !['accuracy', 'technical performance', 'cost efficiency'].includes(k.toLowerCase()))
                  .slice(0, 5)
                  .map(([k, v]) => (
                    <div key={k} className="rounded-lg border bg-white p-4">
                      <div className="text-xs uppercase tracking-wide text-gray-500">{k}</div>
                      <div className="mt-1 text-2xl font-semibold text-gray-900">
                        {typeof v === 'number' ? v.toFixed(3) : String(v)}
                      </div>
                    </div>
                  ))}
              </>
            ) : (
              <div className="sm:col-span-2 lg:col-span-4 text-sm text-gray-600">
                No summary metrics available yet for the selected experiment.
              </div>
            )}
          </>
        )}
      </section>

      {/* Scenario table */}
      <section className="mt-6">
        <h2 className="text-lg font-semibold text-gray-900">Scenario Results</h2>
        {loadingReport ? (
          <div className="mt-3 space-y-2">
            <Skeleton variant="rect" height={48} />
            <Skeleton variant="rect" height={48} />
            <Skeleton variant="rect" height={48} />
          </div>
        ) : report?.scenario_reports?.length ? (
          <div className="mt-3">
            <Table
              aria-label="Scenario Results"
              columns={scenarioColumns}
              data={report.scenario_reports}
              getRowKey={(r) => r.scenario_key}
            />
          </div>
        ) : (
          <p className="mt-2 text-sm text-gray-600">No scenario results.</p>
        )}
      </section>
    </section>
  );
};

export default Results;