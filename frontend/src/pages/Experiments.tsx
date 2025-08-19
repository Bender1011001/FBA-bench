import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import Skeleton from '../components/ui/Skeleton';
import ExperimentForm, { type ExperimentFormValues } from '../components/experiments/ExperimentForm';
import ExperimentList from '../components/experiments/ExperimentList';
import { createExperiment, getExperiments, RouteNotAvailableError, startSimulation } from '../services/api';
import type { Experiment, Simulation } from '../types/api';
import { connectRealtime, SIMULATION_TOPIC_PREFIX } from '../services/realtime';
import useToast from '../contexts/useToast';

const Experiments: React.FC = () => {
  const toast = useToast();
  const [items, setItems] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [openCreate, setOpenCreate] = useState(false);
  const [liveStatuses, setLiveStatuses] = useState<Record<string, Experiment['status']>>({});
  const [startingId, setStartingId] = useState<string | null>(null);

  const realtimeRef = useRef<ReturnType<typeof connectRealtime> | null>(null);
  const subscribed = useRef<Set<string>>(new Set());

  const load = async (signal?: AbortSignal) => {
    try {
      setLoading(true);
      const data = await getExperiments(signal);
      setItems(Array.isArray(data) ? data : []);
    } catch (e) {
      if (e instanceof RouteNotAvailableError) {
        toast.error('Experiments API is not available (404). Ensure backend routes are enabled.', 5000);
      } else {
        const msg = e instanceof Error ? e.message : 'Failed to load experiments.';
        toast.error(msg, 4000);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    // initial fetch
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    load(controller.signal);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    // Init realtime client once
    if (!realtimeRef.current) {
      realtimeRef.current = connectRealtime();
    }
    return () => {
      if (realtimeRef.current) {
        // Unsubscribe all on unmount
        for (const t of subscribed.current) {
          realtimeRef.current.unsubscribe(t);
        }
        realtimeRef.current.close();
        realtimeRef.current = null;
      }
    };
  }, []);

  const handleCreate = async (values: ExperimentFormValues) => {
    try {
      const exp = await createExperiment(values);
      toast.success('Experiment created', 3000);
      setItems((prev) => [exp, ...prev]);
    } catch (e) {
      if (e instanceof RouteNotAvailableError) {
        toast.error('Create experiment route not available (404). Update backend or use mock-friendly UI.', 5000);
      } else {
        const msg = e instanceof Error ? e.message : 'Failed to create experiment.';
        toast.error(msg, 4000);
      }
      throw e;
    }
  };

  const handleStart = async (experiment: Experiment): Promise<Simulation | void> => {
    try {
      setStartingId(experiment.id);
      const sim = await startSimulation(experiment.id);
      toast.success(`Simulation started for "${experiment.name}"`, 3000);

      // subscribe for live simulation updates for this experiment
      const topic = `${SIMULATION_TOPIC_PREFIX}${experiment.id}`;
      if (realtimeRef.current && !subscribed.current.has(topic)) {
        subscribed.current.add(topic);
        realtimeRef.current.subscribe(topic, (msg: unknown) => {
          try {
            // Expect messages with at least { status } or { event }
            const payload = msg as { status?: Experiment['status']; event?: string } | null;
            const status =
              payload?.status ??
              (payload?.event === 'simulation.completed'
                ? 'completed'
                : payload?.event === 'simulation.started'
                ? 'running'
                : undefined);

            if (status) {
              setLiveStatuses((prev) => ({ ...prev, [experiment.id]: status }));
            }
          } catch {
            // ignore bad frames
          }
        });
      }

      // optimistic status
      setLiveStatuses((prev) => ({ ...prev, [experiment.id]: 'running' }));
      return sim;
    } catch (e) {
      if (e instanceof RouteNotAvailableError) {
        toast.error('Start simulation route not available (404). Ensure backend supports /simulations.', 5000);
      } else {
        const msg = e instanceof Error ? e.message : 'Failed to start simulation.';
        toast.error(msg, 4000);
      }
    } finally {
      setStartingId(null);
    }
  };

  const headerActions = useMemo(
    () => (
      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => setOpenCreate(true)}
          className="px-3 py-2 rounded-md bg-blue-600 text-white text-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          aria-label="Create new experiment"
        >
          New Experiment
        </button>
        <Link
          to="/results"
          className="px-3 py-2 rounded-md border text-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
          aria-label="Go to results"
        >
          View Results
        </Link>
        <button
          type="button"
          onClick={() => load()}
          className="px-3 py-2 rounded-md border text-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
          aria-label="Refresh experiments"
        >
          Refresh
        </button>
      </div>
    ),
    [],
  );

  return (
    <section aria-labelledby="experiments-title" className="w-full h-full">
      <header className="mb-4">
        <h1 id="experiments-title" className="text-2xl font-semibold text-gray-900">
          Experiments
        </h1>
        <p className="text-gray-600">Manage experiment definitions, runs, and statuses.</p>
        {headerActions}
      </header>

      <div className="mt-4">
        {loading ? (
          <div className="space-y-2">
            <Skeleton variant="rect" height={48} />
            <Skeleton variant="rect" height={48} />
            <Skeleton variant="rect" height={48} />
          </div>
        ) : (
          <ExperimentList items={items} onStart={handleStart} liveStatuses={liveStatuses} isStartingId={startingId} />
        )}
      </div>

      <ExperimentForm open={openCreate} onClose={() => setOpenCreate(false)} onSubmit={handleCreate} />
    </section>
  );
};

export default Experiments;