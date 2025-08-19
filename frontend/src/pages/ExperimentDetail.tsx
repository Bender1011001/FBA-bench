import React, { useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import Spinner from '../components/ui/Spinner';
import Skeleton from '../components/ui/Skeleton';
import Badge from '../components/ui/Badge';
import Table, { type Column } from '../components/ui/Table';
import { getEngineReport, getExperiment } from '../services/api';
import type { EngineReport, Experiment } from '../types/api';
import { connectRealtime, EXPERIMENT_TOPIC_PREFIX } from '../services/realtime';
import useToast from '../contexts/useToast';

const MetricCard: React.FC<{ label: string; value: number | string }> = ({ label, value }) => (
  <div className="rounded-lg border bg-white p-4">
    <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
    <div className="mt-1 text-2xl font-semibold text-gray-900">
      {typeof value === 'number' ? value : String(value)}
    </div>
  </div>
);

const ValidatorsCell: React.FC<{ validators: Record<string, boolean | string> }> = ({ validators }) => {
  const entries = Object.entries(validators || {});
  if (!entries.length) return <span className="text-gray-500 text-sm">—</span>;
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
};

const ExperimentDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const toast = useToast();
  const [exp, setExp] = useState<Experiment | null>(null);
  const [report, setReport] = useState<EngineReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingReport, setLoadingReport] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadExp = async (expId: string, signal?: AbortSignal) => {
    const data = await getExperiment(expId, signal);
    setExp(data);
    return data;
  };

  const loadReport = async (expId: string, signal?: AbortSignal) => {
    try {
      const rep = await getEngineReport(expId, signal);
      setReport(rep);
    } catch {
      setReport(null);
      // Soft-fail when report is not yet available
    }
  };

  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    (async () => {
      try {
        await loadExp(id, controller.signal);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Failed to load experiment.';
        setError(msg);
      } finally {
        setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [id]);

  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    setLoadingReport(true);
    (async () => {
      await loadReport(id, controller.signal);
      setLoadingReport(false);
    })();
    return () => controller.abort();
  }, [id]);

  // Realtime: listen to experiment:<id> events -> refetch when completed/updated
  useEffect(() => {
    if (!id) return;
    const client = connectRealtime();
    const topic = `${EXPERIMENT_TOPIC_PREFIX}${id}`;
    client.subscribe(topic, (msg: unknown) => {
      try {
        const maybe = msg as { event?: string } | null;
        const ev = maybe?.event;
        if (ev === 'experiment.updated' || ev === 'experiment.completed') {
          // Refetch experiment and possibly report
          (async () => {
            try {
              await loadExp(id);
              await loadReport(id);
              if (ev === 'experiment.completed') {
                toast.success('Experiment completed. Report refreshed.', 4000);
              }
            } catch {
              // ignore
            }
          })();
        }
      } catch {
        // ignore malformed messages
      }
    });
    return () => {
      client.unsubscribe(topic);
      client.close();
    };
  }, [id, toast]);

  const totals = useMemo(() => report?.totals ?? {}, [report?.totals]);

  const scenarioColumns: Column<NonNullable<EngineReport>['scenario_reports'][number]>[] = [
    { key: 'scenario_key', header: 'Scenario' },
    {
      key: 'metrics',
      header: 'Metrics',
      render: (row) => {
        const m = row.metrics || {};
        const entries = Object.entries(m);
        if (!entries.length) return <span className="text-gray-500 text-sm">—</span>;
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
      render: (row) => <ValidatorsCell validators={row.validators || {}} />,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (row) => <span className="text-sm text-gray-700">{row.summary || '—'}</span>,
    },
  ];

  if (!id) {
    return (
      <section className="w-full h-full">
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">Missing experiment ID.</div>
      </section>
    );
  }

  if (loading) {
    return (
      <section className="w-full h-full">
        <div className="flex items-center justify-center h-64">
          <Spinner size="lg" label="Loading experiment…" />
        </div>
      </section>
    );
  }

  if (error || !exp) {
    return (
      <section className="w-full h-full">
        <div className="mb-3">
          <Link to="/experiments" className="text-blue-700 hover:underline">
            ← Back to Experiments
          </Link>
        </div>
        <div role="alert" className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
          {error ?? 'Experiment not found.'}
        </div>
      </section>
    );
  }

  const statusTone: React.ComponentProps<typeof Badge>['tone'] =
    exp.status === 'running' ? 'info' : exp.status === 'completed' ? 'success' : exp.status === 'failed' ? 'danger' : 'neutral';

  return (
    <section aria-labelledby="experiment-detail-title" className="w-full h-full">
      <div className="mb-3">
        <Link to="/experiments" className="text-blue-700 hover:underline">
          ← Back to Experiments
        </Link>
      </div>

      <header className="mb-4">
        <h1 id="experiment-detail-title" className="text-2xl font-semibold text-gray-900">
          {exp.name}{' '}
          <Badge tone={statusTone} aria-label={`Status ${exp.status}`}>
            {exp.status}
          </Badge>
        </h1>
        <p className="text-gray-600">{exp.description || 'No description provided.'}</p>
        <div className="text-xs text-gray-500 mt-1">
          Created {new Date(exp.created_at).toLocaleString()} · Updated {new Date(exp.updated_at).toLocaleString()}
        </div>
      </header>

      {/* Totals / Key metrics */}
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
            {Object.keys(totals).length === 0 ? (
              <div className="sm:col-span-2 lg:col-span-4 text-sm text-gray-600">
                No summary metrics available yet. If the run is still in progress, refresh when completed.
              </div>
            ) : (
              Object.entries(totals).slice(0, 8).map(([name, value]) => (
                <MetricCard key={name} label={name} value={typeof value === 'number' ? Number(value.toFixed(3)) : String(value)} />
              ))
            )}
          </>
        )}
      </section>

      {/* Scenario reports */}
      <section className="mt-6">
        <h2 className="text-lg font-semibold text-gray-900">Scenario Reports</h2>
        {loadingReport ? (
          <div className="mt-3 space-y-2">
            <Skeleton variant="rect" height={48} />
            <Skeleton variant="rect" height={48} />
            <Skeleton variant="rect" height={48} />
          </div>
        ) : report?.scenario_reports?.length ? (
          <div className="mt-3">
            <Table
              aria-label="Scenario Reports"
              columns={scenarioColumns}
              data={report.scenario_reports}
              getRowKey={(r) => r.scenario_key}
            />
          </div>
        ) : (
          <p className="mt-2 text-sm text-gray-600">No scenarios reported.</p>
        )}
      </section>
    </section>
  );
};

export default ExperimentDetail;