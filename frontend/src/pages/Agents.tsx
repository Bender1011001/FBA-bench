import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import Skeleton from '../components/ui/Skeleton';
import Table, { type Column } from '../components/ui/Table';
import AgentForm, { type AgentFormValues } from '../components/agents/AgentForm';
import { createAgent, deleteAgent, getAgents, RouteNotAvailableError, updateAgent } from '../services/api';
import type { Agent } from '../types/api';
import useToast from '../contexts/useToast';

const Agents: React.FC = () => {
  const toast = useToast();
  const [items, setItems] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [openCreate, setOpenCreate] = useState(false);
  const [editing, setEditing] = useState<Agent | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = async (signal?: AbortSignal) => {
    try {
      setLoading(true);
      const data = await getAgents(signal);
      setItems(Array.isArray(data) ? data : []);
    } catch (e) {
      if (e instanceof RouteNotAvailableError) {
        toast.error('Agents API is not available (404). Ensure backend routes are enabled.', 5000);
      } else {
        const msg = e instanceof Error ? e.message : 'Failed to load agents.';
        toast.error(msg, 4000);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    load(controller.signal);
    return () => controller.abort();
  }, []);

  const onCreate = async (values: AgentFormValues) => {
    try {
      const created = await createAgent(values);
      setItems((prev) => [created, ...prev]);
      toast.success('Agent created', 3000);
    } catch (e) {
      if (e instanceof RouteNotAvailableError) {
        toast.error('Create agent route not available (404).', 5000);
      } else {
        const msg = e instanceof Error ? e.message : 'Failed to create agent.';
        toast.error(msg, 4000);
      }
      throw e;
    }
  };

  const onUpdate = async (values: AgentFormValues) => {
    if (!editing) return;
    try {
      setBusyId(editing.id);
      // optimistic update
      const nextLocal: Agent = {
        ...editing,
        name: values.name,
        runner: values.runner,
        description: values.description,
        config: values.config,
        updated_at: new Date().toISOString(),
      };
      setItems((list) => list.map((a) => (a.id === editing.id ? nextLocal : a)));

      const updated = await updateAgent(editing.id, values);
      setItems((list) => list.map((a) => (a.id === editing.id ? updated : a)));
      toast.success('Agent updated', 3000);
      setEditing(null);
    } catch (e) {
      // revert optimistic
      await load();
      if (e instanceof RouteNotAvailableError) {
        toast.error('Update agent route not available (404).', 5000);
      } else {
        const msg = e instanceof Error ? e.message : 'Failed to update agent.';
        toast.error(msg, 4000);
      }
      throw e;
    } finally {
      setBusyId(null);
    }
  };

  const onDelete = async (agent: Agent) => {
    if (!window.confirm(`Delete agent "${agent.name}"? This action cannot be undone.`)) return;
    try {
      setBusyId(agent.id);
      // optimistic remove
      setItems((list) => list.filter((a) => a.id !== agent.id));
      await deleteAgent(agent.id);
      toast.success('Agent deleted', 3000);
    } catch (e) {
      await load();
      if (e instanceof RouteNotAvailableError) {
        toast.error('Delete agent route not available (404).', 5000);
      } else {
        const msg = e instanceof Error ? e.message : 'Failed to delete agent.';
        toast.error(msg, 4000);
      }
    } finally {
      setBusyId(null);
    }
  };

  const columns: Column<Agent>[] = useMemo(
    () => [
      { key: 'name', header: 'Name' },
      { key: 'runner', header: 'Runner', className: 'whitespace-nowrap' },
      {
        key: 'description',
        header: 'Description',
        render: (row) => <span className="text-sm text-gray-700">{row.description || '—'}</span>,
      },
      {
        key: 'updated_at',
        header: 'Updated',
        render: (row) => new Date(row.updated_at).toLocaleString(),
        className: 'whitespace-nowrap',
      },
      {
        key: 'actions',
        header: 'Actions',
        render: (row) => {
          const disabled = busyId === row.id;
          return (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setEditing(row)}
                className="inline-flex items-center rounded-md border px-2 py-1 text-xs hover:bg-gray-50"
                disabled={disabled}
              >
                Edit
              </button>
              <button
                type="button"
                onClick={() => onDelete(row)}
                className="inline-flex items-center rounded-md bg-red-600 px-2 py-1 text-xs text-white disabled:opacity-60"
                disabled={disabled}
              >
                Delete
              </button>
            </div>
          );
        },
      },
    ],
    [busyId],
  );

  return (
    <section aria-labelledby="agents-title" className="w-full h-full">
      <header className="mb-4">
        <h1 id="agents-title" className="text-2xl font-semibold text-gray-900">
          Agents
        </h1>
        <p className="text-gray-600">View, configure, and manage agents.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setOpenCreate(true)}
            className="px-3 py-2 rounded-md bg-blue-600 text-white text-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            aria-label="Register new agent"
          >
            Register Agent
          </button>
          <Link
            to="/experiments"
            className="px-3 py-2 rounded-md border text-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
            aria-label="Go to experiments"
          >
            Manage Experiments
          </Link>
          <button
            type="button"
            onClick={() => load()}
            className="px-3 py-2 rounded-md border text-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
            aria-label="Refresh agents"
          >
            Refresh
          </button>
        </div>
      </header>

      <div className="mt-4">
        {loading ? (
          <div className="space-y-2">
            <Skeleton variant="rect" height={48} />
            <Skeleton variant="rect" height={48} />
            <Skeleton variant="rect" height={48} />
          </div>
        ) : items.length ? (
          <Table aria-label="Agents" columns={columns} data={items} getRowKey={(r) => r.id} />
        ) : (
          <div className="rounded border bg-white p-4">
            <div className="text-sm text-gray-700">No agents found.</div>
            <button
              type="button"
              onClick={() => setOpenCreate(true)}
              className="mt-2 px-3 py-2 rounded-md bg-blue-600 text-white text-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Create Agent
            </button>
          </div>
        )}
      </div>

      <AgentForm open={openCreate} onClose={() => setOpenCreate(false)} onSubmit={onCreate} title="Create Agent" />
      <AgentForm
        open={Boolean(editing)}
        onClose={() => setEditing(null)}
        onSubmit={onUpdate}
        initial={editing ?? undefined}
        title="Edit Agent"
      />
    </section>
  );
};

export default Agents;