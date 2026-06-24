import React, { useState, useEffect } from 'react';
import {
  Play,
  Activity,
  Settings,
  BookOpen,
  Layers,
  XCircle,
  RefreshCw,
  Sliders,
  Info,
  Download,
  Shield,
  Database,
  Cpu,
  ChevronRight,
  TrendingUp,
  Award,
  CheckCircle2,
  AlertOctagon,
  HelpCircle,
  Edit3,
  AlignLeft,
  Settings2,
  Zap,
  BarChart2,
  Brain,
  Sparkles,
} from 'lucide-react';
import type { GradingResult, Weights, BatchItem, SystemThresholds } from './types';
import {
  DEFAULT_WEIGHTS,
  WEIGHT_PRESETS,
  DEFAULT_THRESHOLDS,
  simulateGrading,
  calculateLocalGrade,
} from './mockData';

// ─── Helpers ─────────────────────────────────────────────────────────────────

const getScoreColor = (score: number) => {
  if (score >= 80) return { stroke: 'stroke-emerald-500', text: 'text-emerald-600', bg: 'bg-emerald-500' };
  if (score >= 55) return { stroke: 'stroke-amber-500', text: 'text-amber-600', bg: 'bg-amber-500' };
  return { stroke: 'stroke-rose-500', text: 'text-rose-600', bg: 'bg-rose-500' };
};

const getScoreLabel = (score: number) => {
  if (score >= 80) return { label: 'Excellent', color: 'text-emerald-700 bg-emerald-50 border-emerald-200' };
  if (score >= 55) return { label: 'Partial', color: 'text-amber-700 bg-amber-50 border-amber-200' };
  return { label: 'Needs Work', color: 'text-rose-700 bg-rose-50 border-rose-200' };
};

const getBadgeStyle = (tag: string) => {
  switch (tag) {
    case 'Correct':           return 'bg-emerald-100 text-emerald-900 border-2 border-emerald-400';
    case 'Partially Correct': return 'bg-amber-100 text-amber-900 border-2 border-amber-400';
    case 'Incorrect':         return 'bg-rose-100 text-rose-900 border-2 border-rose-400';
    case 'Missing Concepts':  return 'bg-sky-100 text-sky-900 border-2 border-sky-400';
    case 'Factual Error':     return 'bg-red-100 text-red-900 border-2 border-red-400';
    case 'Logical Error':     return 'bg-purple-100 text-purple-900 border-2 border-purple-400';
    case 'Vague Expression':  return 'bg-slate-200 text-slate-800 border-2 border-slate-400';
    case 'Grammar Error':     return 'bg-orange-100 text-orange-900 border-2 border-orange-400';
    case 'Off-Topic':         return 'bg-fuchsia-100 text-fuchsia-900 border-2 border-fuchsia-400';
    case 'Incomplete':        return 'bg-yellow-100 text-yellow-900 border-2 border-yellow-400';
    default:                  return 'bg-slate-100 text-slate-700 border-2 border-slate-300';
  }
};

const getTagIcon = (tag: string) => {
  switch (tag) {
    case 'Correct':           return <CheckCircle2 className="w-3 h-3 text-emerald-700 flex-shrink-0" strokeWidth={2.5} />;
    case 'Partially Correct': return <Info className="w-3 h-3 text-amber-700 flex-shrink-0" strokeWidth={2.5} />;
    case 'Incorrect':         return <XCircle className="w-3 h-3 text-rose-700 flex-shrink-0" strokeWidth={2.5} />;
    case 'Factual Error':
    case 'Logical Error':     return <AlertOctagon className="w-3 h-3 text-red-700 flex-shrink-0" strokeWidth={2.5} />;
    default:                  return <HelpCircle className="w-3 h-3 text-slate-500 flex-shrink-0" strokeWidth={2} />;
  }
};

// ─── Sub-components ───────────────────────────────────────────────────────────

/** Circular SVG score ring */
function ScoreRing({ score, size = 96 }: { score: number; size?: number }) {
  const c = getScoreColor(score);
  const r = 15.9155;
  const circ = 283;
  const offset = circ - (circ * score) / 100;
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="w-full h-full" viewBox="0 0 36 36">
        <path
          strokeWidth="2.5" fill="none" stroke="#e2e8f0"
          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
        />
        <path
          className={`progress-circle ${c.stroke}`}
          strokeWidth="2.5" strokeLinecap="round" fill="none"
          style={{ '--offset': offset } as React.CSSProperties}
          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
        />
      </svg>
      <div className={`absolute inset-0 flex flex-col items-center justify-center ${c.text}`}>
        <span className="text-lg font-semibold tabular-nums leading-none">{Math.round(score)}</span>
        <span className="text-[9px] text-slate-500 leading-none mt-0.5">/ 100</span>
      </div>
    </div>
  );
}

/** Horizontal metric bar */
function MetricBar({
  label, value, color,
}: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <span className="text-xs text-slate-500">{label}</span>
        <span className="text-xs font-semibold text-slate-800 tabular-nums">{Math.round(value)}%</span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full metric-bar-fill ${color}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

/** Tag badge chip */
function Badge({ tag }: { tag: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-semibold ${getBadgeStyle(tag)}`}>
      {getTagIcon(tag)}
      {tag}
    </span>
  );
}

/** Empty state placeholder */
function EmptyState({ icon: Icon, title, body }: { icon: React.ElementType; title: string; body: string }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-14 space-y-3">
      <div className="w-12 h-12 rounded-2xl bg-slate-100 border border-slate-200 flex items-center justify-center">
        <Icon className="w-5 h-5 text-slate-400" strokeWidth={1.5} />
      </div>
      <p className="text-sm font-semibold text-slate-800">{title}</p>
      <p className="text-xs text-slate-500 max-w-xs leading-relaxed">{body}</p>
    </div>
  );
}

/** Section card wrapper */
function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white rounded-2xl border border-slate-200/80 shadow-sm ${className}`}>
      {children}
    </div>
  );
}

/** Section title row inside a card */
function CardHeader({ icon: Icon, title, right }: {
  icon: React.ElementType; title: string; right?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between px-6 pt-5 pb-4 border-b border-slate-100">
      <div className="flex items-center gap-2.5">
        <div className="w-7 h-7 rounded-lg bg-slate-100 flex items-center justify-center">
          <Icon className="w-3.5 h-3.5 text-slate-500" strokeWidth={2} />
        </div>
        <span className="text-sm font-semibold text-slate-800">{title}</span>
      </div>
      {right && <div>{right}</div>}
    </div>
  );
}

// ─── Stats Card ───────────────────────────────────────────────────────────────

function StatCard({
  label, value, sub, accent,
}: { label: string; value: string | number; sub: string; accent: string }) {
  return (
    <Card className="card-hover p-6">
      <p className={`text-[10px] font-semibold uppercase tracking-widest ${accent}`}>{label}</p>
      <p className="text-3xl font-semibold text-slate-800 mt-1.5 tabular-nums">{value}</p>
      <p className="text-xs text-slate-500 mt-1">{sub}</p>
    </Card>
  );
}

// ─── Weight Sliders Panel ─────────────────────────────────────────────────────

function WeightsPanel({
  weights, activePreset, onPresetChange, onWeightChange,
}: {
  weights: Weights;
  activePreset: string;
  onPresetChange: (p: string) => void;
  onWeightChange: (k: keyof Weights, v: number) => void;
}) {
  const total = Object.values(weights).reduce((a, b) => a + b, 0);
  const isBalanced = Math.abs(total - 1) < 0.01;

  const presets = [
    { key: 'balanced', label: 'Balanced' },
    { key: 'content_focused', label: 'Content' },
    { key: 'academic_writing', label: 'Academic' },
    { key: 'logic_heavy', label: 'Logic' },
    { key: 'quick_check', label: 'Quick' },
  ];

  const sliders: { key: keyof Weights; label: string; color: string }[] = [
    { key: 'semantic',  label: 'Semantic Similarity', color: 'bg-indigo-500' },
    { key: 'coverage',  label: 'Keyword Coverage',    color: 'bg-sky-500' },
    { key: 'formality', label: 'Formality',           color: 'bg-slate-500' },
    { key: 'grammar',   label: 'Grammar',             color: 'bg-orange-500' },
    { key: 'logic',     label: 'Logical Coherence',   color: 'bg-purple-500' },
  ];

  return (
    <Card>
      <CardHeader icon={Settings2} title="Weights Configuration" />
      <div className="p-5 space-y-5">
        {/* Preset pills */}
        <div className="flex flex-wrap gap-1.5">
          {presets.map((p) => (
            <button
              key={p.key}
              onClick={() => onPresetChange(p.key)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 ${
                activePreset === p.key
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {p.label}
            </button>
          ))}
          {activePreset === 'custom' && (
            <span className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-violet-100 text-violet-700 border border-violet-200">
              Custom
            </span>
          )}
        </div>

        {/* Sliders */}
        <div className="space-y-4">
          {sliders.map(({ key, label, color }) => (
            <div key={key} className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-600 font-semibold">{label}</span>
                <span className="font-semibold text-slate-800 tabular-nums">{Math.round(weights[key] * 100)}%</span>
              </div>
              <input
                type="range" min="0" max="1" step="0.05"
                value={weights[key]}
                onChange={(e) => onWeightChange(key, parseFloat(e.target.value))}
                className="w-full"
                style={{ background: `linear-gradient(to right, ${color === 'bg-indigo-500' ? '#6366f1' : color === 'bg-sky-500' ? '#0ea5e9' : color === 'bg-slate-500' ? '#64748b' : color === 'bg-orange-500' ? '#f97316' : '#a855f7'} ${weights[key] * 100}%, #e2e8f0 ${weights[key] * 100}%)` }}
              />
            </div>
          ))}
        </div>

        {/* Total indicator */}
        <div className={`flex items-center gap-2 text-xs px-3 py-2.5 rounded-xl border ${
          isBalanced
            ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
            : 'bg-amber-50 border-amber-200 text-amber-700'
        }`}>
          <Info className="w-3.5 h-3.5 flex-shrink-0" strokeWidth={2} />
          <span>
            Total: <span className="font-semibold tabular-nums">{Math.round(total * 100)}%</span>
            {isBalanced ? ' · Weights auto-normalize on evaluation.' : ' · Weights will be normalized automatically.'}
          </span>
        </div>
      </div>
    </Card>
  );
}

// ─── Single Result Panel ──────────────────────────────────────────────────────

function ResultPanel({ result }: { result: GradingResult }) {
  const score = result.metrics.final_grade;
  const sl = getScoreLabel(score);

  return (
    <div className="space-y-4 animate-fade-up">
      {/* Score hero card */}
      <Card className="p-6">
        <div className="flex items-center gap-6">
          <ScoreRing score={score} size={88} />
          <div className="flex-1 space-y-2.5">
            <div className="flex items-center gap-2">
              <span className={`px-2.5 py-1 rounded-lg border text-xs font-semibold ${sl.color}`}>
                {sl.label}
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">Weighted composite score across 5 NLP criteria.</p>
            <div className="flex flex-wrap gap-1.5">
              {result.feedback.tags.map((tag, i) => <Badge key={i} tag={tag} />)}
            </div>
          </div>
        </div>
      </Card>

      {/* Metric bars */}
      <Card>
        <CardHeader icon={BarChart2} title="Feature Breakdown" />
        <div className="p-5 space-y-3.5">
          <MetricBar label="Semantic Similarity" value={result.metrics.semantic_score * 100}  color="bg-indigo-500" />
          <MetricBar label="Keyword Coverage"    value={result.metrics.coverage_score * 100}   color="bg-sky-500" />
          <MetricBar label="Formality"           value={result.metrics.formality_score * 100} color="bg-slate-400" />
          <MetricBar label="Grammar"             value={result.metrics.grammar_score * 100}    color="bg-orange-500" />
          <MetricBar label="Logical Coherence"   value={result.metrics.logic_score * 100}      color="bg-purple-500" />
        </div>
      </Card>

      {/* LLM Feedback */}
      <Card>
        <CardHeader icon={Brain} title="AI Feedback" />
        <div className="p-5 space-y-4">
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Explanation</p>
            <p className="text-xs text-slate-700 leading-relaxed">{result.feedback.explanation}</p>
          </div>
          <div className="space-y-1.5 pt-3 border-t border-slate-100">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Recommendation</p>
            <p className="text-xs text-slate-700 leading-relaxed">{result.feedback.suggestion}</p>
          </div>
        </div>
      </Card>

      {/* Missing keywords */}
      {result.metrics.missing_keywords?.length > 0 && (
        <Card>
          <div className="p-5 space-y-2.5">
            <p className="text-[10px] font-semibold text-rose-600 uppercase tracking-widest">Missing Keywords</p>
            <div className="flex flex-wrap gap-1.5">
              {result.metrics.missing_keywords.map((kw, i) => (
                <span key={i} className="px-2.5 py-1 rounded-lg bg-rose-50 text-rose-800 border border-rose-200 text-[10px] font-semibold">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* NLI details */}
      {result.metrics.logic_details && (
        <Card>
          <div className="p-5 space-y-2.5">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">NLI Probabilities</p>
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs text-slate-600">
              {Object.entries(result.metrics.logic_details).map(([k, v]) =>
                v !== undefined ? (
                  <div key={k} className="flex justify-between border-b border-slate-100 pb-1.5">
                    <span className="capitalize">{k.replace(/_/g, ' ')}</span>
                    <span className="font-semibold tabular-nums text-slate-800">{Math.round((v as number) * 100)}%</span>
                  </div>
                ) : null
              )}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'single' | 'batch' | 'settings'>('dashboard');
  const [isLive, setIsLive] = useState<boolean>(false);
  const [apiUrl, setApiUrl] = useState<string>('http://localhost:8000');
  const [healthStatus, setHealthStatus] = useState<{ status: string; device: string; models_loaded: boolean } | null>(null);

  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const [activePreset, setActivePreset] = useState<string>('balanced');
  const [thresholds] = useState<SystemThresholds>(DEFAULT_THRESHOLDS);

  const [singleContext, setSingleContext]   = useState<string>('Photosynthesis is the process by which plants make food.');
  const [singleQuestion, setSingleQuestion] = useState<string>('Explain the process of photosynthesis.');
  const [singleReference, setSingleReference] = useState<string>('Photosynthesis converts sunlight, CO2, and water into glucose and oxygen using chlorophyll.');
  const [singleStudent, setSingleStudent]   = useState<string>('Plants use sunlight and water to make food and release oxygen.');
  const [singleResult, setSingleResult]     = useState<GradingResult | null>(null);
  const [isGradingSingle, setIsGradingSingle] = useState<boolean>(false);
  const [singleError, setSingleError]       = useState<string | null>(null);

  const [batchQuestion, setBatchQuestion]   = useState<string>('Explain the process of photosynthesis.');
  const [batchReference, setBatchReference] = useState<string>('Photosynthesis converts sunlight, CO2, and water into glucose and oxygen using chlorophyll.');
  const [batchRawInput, setBatchRawInput]   = useState<string>(
    `Student A: Plants use light to make food.\nStudent B: Animals eat plants to get energy.\nStudent C: Plants use sunlight, water, carbon dioxide to produce glucose and oxygen via chlorophyll.\nStudent D: It is a process.`
  );
  const [batchItems, setBatchItems]           = useState<BatchItem[]>([]);
  const [isGradingBatch, setIsGradingBatch]   = useState<boolean>(false);
  const [selectedBatchItem, setSelectedBatchItem] = useState<BatchItem | null>(null);

  const [stats, setStats] = useState({
    totalGraded: 4, averageGrade: 64.63, correctCount: 1, partialCount: 2, incorrectCount: 1,
  });

  useEffect(() => {
    if (isLive) checkHealth();
    else setHealthStatus({ status: 'healthy', device: 'Client Sim', models_loaded: true });
  }, [isLive, apiUrl]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${apiUrl}/health`);
      if (res.ok) setHealthStatus(await res.json());
      else setHealthStatus({ status: 'offline', device: 'None', models_loaded: false });
    } catch {
      setHealthStatus({ status: 'offline', device: 'None', models_loaded: false });
    }
  };

  const handlePresetChange = (name: string) => {
    setActivePreset(name);
    if (WEIGHT_PRESETS[name]) {
      setWeights(WEIGHT_PRESETS[name]);
      if (singleResult) recalculateSingle(WEIGHT_PRESETS[name]);
    }
  };

  const handleWeightChange = (key: keyof Weights, value: number) => {
    setActivePreset('custom');
    const nw = { ...weights, [key]: Math.round(value * 100) / 100 };
    setWeights(nw);
    if (singleResult) recalculateSingle(nw);
  };

  const recalculateSingle = async (nw: Weights) => {
    if (!singleResult) return;
    if (isLive) {
      try {
        const res = await fetch(`${apiUrl}/grade/recalculate`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ metrics: singleResult.metrics, weights: nw }),
        });
        if (res.ok) {
          const data = await res.json();
          setSingleResult(prev => prev ? { ...prev, metrics: { ...prev.metrics, final_grade: data.new_grade } } : null);
        }
      } catch {
        const g = calculateLocalGrade(singleResult.metrics, nw);
        setSingleResult(prev => prev ? { ...prev, metrics: { ...prev.metrics, final_grade: g } } : null);
      }
    } else {
      const g = calculateLocalGrade(singleResult.metrics, nw);
      setSingleResult(prev => prev ? { ...prev, metrics: { ...prev.metrics, final_grade: g } } : null);
    }
  };

  const handleGradeSingle = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGradingSingle(true);
    setSingleError(null);
    if (isLive) {
      try {
        const res = await fetch(`${apiUrl}/grade`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ context: singleContext, question: singleQuestion, reference: singleReference, student: singleStudent, weights }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.success) { setSingleResult(data.data); updateStats(data.data.metrics.final_grade, data.data.feedback.tags); }
          else setSingleError(data.error || 'Unknown error.');
        } else {
          const e2 = await res.json().catch(() => ({}));
          setSingleError(e2.error || `Server error ${res.status}`);
        }
      } catch { setSingleError('Cannot reach backend. Switch to Simulated Mode or start uvicorn.'); }
      finally { setIsGradingSingle(false); }
    } else {
      setTimeout(() => {
        try {
          const r = simulateGrading(singleContext, singleQuestion, singleReference, singleStudent, weights);
          setSingleResult(r); updateStats(r.metrics.final_grade, r.feedback.tags);
        } catch (err: any) { setSingleError(err.message || 'Simulation error'); }
        finally { setIsGradingSingle(false); }
      }, 800);
    }
  };

  const updateStats = (score: number, tags: string[]) => {
    setStats(prev => {
      const nt = prev.totalGraded + 1;
      return {
        totalGraded:    nt,
        averageGrade:   Math.round(((prev.averageGrade * prev.totalGraded + score) / nt) * 100) / 100,
        correctCount:   prev.correctCount   + (tags.includes('Correct') ? 1 : 0),
        partialCount:   prev.partialCount   + (tags.includes('Partially Correct') ? 1 : 0),
        incorrectCount: prev.incorrectCount + (!tags.includes('Correct') && !tags.includes('Partially Correct') ? 1 : 0),
      };
    });
  };

  const handleGradeBatch = async () => {
    if (!batchRawInput.trim()) return;
    setIsGradingBatch(true);
    setSelectedBatchItem(null);
    const lines = batchRawInput.split('\n').filter(l => l.trim());
    const items: BatchItem[] = lines.map((line, idx) => {
      let id = `Student ${idx + 1}`, student = line;
      const ci = line.indexOf(':');
      if (ci > 0 && ci < 30) { id = line.substring(0, ci).trim(); student = line.substring(ci + 1).trim(); }
      return { id, context: '', question: batchQuestion, reference: batchReference, student, status: 'grading' };
    });
    setBatchItems(items);

    if (isLive) {
      try {
        const res = await fetch(`${apiUrl}/grade/batch`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ items: items.map(i => ({ context: i.context, question: i.question, reference: i.reference, student: i.student })), weights }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.success && data.data.results) {
            setBatchItems(prev => prev.map((item, idx) => ({ ...item, status: 'success', result: data.data.results[idx] })));
          } else throw new Error(data.message);
        } else throw new Error(`HTTP ${res.status}`);
      } catch (err: any) {
        setBatchItems(prev => prev.map(item => ({ ...item, status: 'failed', error: err.message })));
      } finally { setIsGradingBatch(false); }
    } else {
      for (let i = 0; i < items.length; i++) {
        await new Promise(r => setTimeout(r, 320));
        const res = simulateGrading('', items[i].question, items[i].reference, items[i].student, weights);
        setBatchItems(prev => { const u = [...prev]; u[i] = { ...u[i], status: 'success', result: res }; return u; });
      }
      setIsGradingBatch(false);
    }
  };

  const handleLoadSample = (topic: 'photosynthesis' | 'watercycle') => {
    if (topic === 'photosynthesis') {
      setBatchQuestion('Explain the process of photosynthesis.');
      setBatchReference('Photosynthesis converts sunlight, CO2, and water into glucose and oxygen using chlorophyll.');
      setBatchRawInput(`Alice: Plants use sunlight, carbon dioxide, and water to make glucose and release oxygen via chlorophyll.\nBob: Plants require sunlight and water to produce food. They release oxygen gas.\nCharlie: Animals eat grass and plants to acquire chemical energy which was generated by sun.\nDavid: water and sun`);
    } else {
      setBatchQuestion('Describe the water cycle.');
      setBatchReference('The water cycle is the continuous process where water evaporates from the Earth, condenses into clouds, and falls as precipitation.');
      setBatchRawInput(`John: Water evaporates from oceans, forms clouds in the sky, and then rains back down to Earth.\nSarah: The water cycle involves evaporation of water, condensation into clouds, and precipitation like rain.\nKevin: Water goes up to the sky because of heat. This happens many times.\nEmma: Rain falls from clouds.`);
    }
  };

  const handleExportCSV = () => {
    if (!batchItems.length) return;
    let csv = 'data:text/csv;charset=utf-8,Student,Answer,Grade,Correctness,Explanation\n';
    batchItems.forEach(item => {
      const ans = `"${item.student.replace(/"/g, '""')}"`;
      const grade = item.result?.metrics.final_grade ?? 'N/A';
      const tag = item.result?.feedback.tags[0] ?? 'N/A';
      const exp = item.result ? `"${item.result.feedback.explanation.replace(/"/g, '""')}"` : 'N/A';
      csv += `${item.id},${ans},${grade},${tag},${exp}\n`;
    });
    const link = document.createElement('a');
    link.setAttribute('href', encodeURI(csv));
    link.setAttribute('download', 'grading_results.csv');
    document.body.appendChild(link); link.click(); document.body.removeChild(link);
  };

  // ─── Nav items ────────────────────────────────────────────────────────────

  const navItems = [
    { id: 'dashboard', label: 'Overview',      icon: Activity },
    { id: 'single',    label: 'Single Grader', icon: Play },
    { id: 'batch',     label: 'Batch Grader',  icon: Layers },
    { id: 'settings',  label: 'Settings',      icon: Settings },
  ] as const;

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans antialiased select-none">

      {/* ══ HEADER ══════════════════════════════════════════════════════════ */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200/70 shadow-sm">
        <div className="flex items-center justify-between px-6 py-3.5">

          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center shadow-sm">
              <Award className="w-4 h-4 text-white" strokeWidth={2.5} />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-800 leading-tight">HybridASAG Grader</h1>
              <p className="text-[10px] text-slate-500 leading-tight">Short Answer Evaluation</p>
            </div>
          </div>

          {/* Right controls */}
          <div className="flex items-center gap-4">

            {/* Mode toggle */}
            <div className="flex items-center bg-slate-100 rounded-xl p-1 gap-0.5">
              <button
                onClick={() => setIsLive(false)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                  !isLive ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                Simulated
              </button>
              <button
                onClick={() => setIsLive(true)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                  isLive ? 'bg-white text-emerald-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                Live API
              </button>
            </div>

            {/* Health status */}
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className={`animate-pulse-dot absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  healthStatus?.status === 'healthy' ? 'bg-emerald-400' : 'bg-rose-400'
                }`} />
                <span className={`relative inline-flex rounded-full h-2 w-2 ${
                  healthStatus?.status === 'healthy' ? 'bg-emerald-500' : 'bg-rose-500'
                }`} />
              </span>
              <span className="text-xs text-slate-500">
                {isLive ? healthStatus?.status ?? 'connecting…' : 'Simulated'}
              </span>
            </div>

          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* ══ SIDEBAR ═════════════════════════════════════════════════════════ */}
        <nav className="w-56 bg-slate-900 border-r border-slate-800 flex flex-col py-5 px-3">
          <div className="space-y-1">
            {navItems.map(({ id, label, icon: Icon }) => {
              const active = activeTab === id;
              return (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={`relative w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-150 ${
                    active
                      ? 'bg-slate-800 text-white'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  {active && <span className="nav-active-indicator" />}
                  <Icon className={`w-4 h-4 ${active ? 'text-indigo-400' : 'text-slate-500'}`} strokeWidth={active ? 2.5 : 2} />
                  {label}
                </button>
              );
            })}
          </div>

          {/* System info */}
          <div className="mt-auto mx-1 bg-slate-950 rounded-xl p-3.5 border border-slate-800 space-y-2">
            <div className="flex items-center gap-2 text-slate-500">
              <Cpu className="w-3.5 h-3.5" strokeWidth={1.5} />
              <span className="text-[10px] font-semibold">{healthStatus?.device || 'Detecting…'}</span>
            </div>
            <div className="flex items-center gap-2 text-slate-500">
              <Database className="w-3.5 h-3.5" strokeWidth={1.5} />
              <span className="text-[10px] font-semibold">Models: {healthStatus?.models_loaded ? 'Ready' : 'Unloaded'}</span>
            </div>
          </div>
        </nav>

        {/* ══ MAIN CONTENT ════════════════════════════════════════════════════ */}
        <main className="flex-1 overflow-y-auto bg-slate-50 p-8">

          {/* ── TAB: DASHBOARD ───────────────────────────────────────────── */}
          {activeTab === 'dashboard' && (
            <div className="max-w-5xl mx-auto space-y-8 animate-fade-up">

              {/* Page header */}
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-slate-800 tracking-tight">Performance Overview</h2>
                  <p className="text-sm text-slate-500 mt-1">Aggregated metrics from the current session.</p>
                </div>
                {!isLive && (
                  <div className="flex items-center gap-2 px-3 py-2 bg-amber-50 border border-amber-200 rounded-xl">
                    <Zap className="w-3.5 h-3.5 text-amber-600" strokeWidth={2.5} />
                    <span className="text-xs font-semibold text-amber-700">Demo Mode — simulated grading</span>
                  </div>
                )}
              </div>

              {/* Stats grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="Total Graded"   value={stats.totalGraded}              sub="Sessions evaluated"  accent="text-indigo-600" />
                <StatCard label="Average Grade"  value={`${stats.averageGrade}%`}        sub="Weighted composite"  accent="text-sky-600" />
                <StatCard label="Correct"        value={stats.correctCount}             sub={`Pass rate ${Math.round((stats.correctCount / stats.totalGraded) * 100 || 0)}%`} accent="text-emerald-600" />
                <StatCard label="Incorrect"      value={stats.incorrectCount}           sub="Capped / penalized"  accent="text-rose-600" />
              </div>

              {/* Pipeline diagram */}
              <Card>
                <CardHeader icon={BookOpen} title="Hybrid ASAG Pipeline" />
                <div className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                    {[
                      { n: '1', title: 'Semantic',   model: 'SimCSE · RoBERTa-Large', color: 'indigo' },
                      { n: '2', title: 'Coverage',   model: 'KeyBERT · MiniLM',        color: 'sky' },
                      { n: '3', title: 'Writing',    model: 'CoLA + RoBERTa',           color: 'orange' },
                      { n: '4', title: 'Logic NLI',  model: 'DeBERTa v3-Base',          color: 'purple' },
                      { n: '5', title: 'Reasoning',  model: 'Qwen2.5-3B-Instruct',      color: 'slate' },
                    ].map((step, i, arr) => (
                      <React.Fragment key={step.n}>
                        <div className={`p-4 rounded-2xl border bg-${step.color}-50/80 border-${step.color}-200 border-t-4 border-t-${step.color}-500 space-y-1.5`}>
                          <span className={`text-[10px] font-semibold text-${step.color}-700 uppercase tracking-wider`}>Step {step.n}</span>
                          <p className={`text-sm font-semibold text-${step.color}-900`}>{step.title}</p>
                          <p className={`text-[10px] text-${step.color}-700`}>{step.model}</p>
                        </div>
                        {i < arr.length - 1 && (
                          <div className="hidden md:flex items-center justify-center text-slate-300">
                            <ChevronRight className="w-4 h-4" strokeWidth={2} />
                          </div>
                        )}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </Card>

            </div>
          )}

          {/* ── TAB: SINGLE GRADER ──────────────────────────────────────── */}
          {activeTab === 'single' && (
            <div className="max-w-6xl mx-auto animate-fade-up">
              <div className="mb-6">
                <h2 className="text-xl font-semibold text-slate-800 tracking-tight">Grade Student Answer</h2>
                <p className="text-sm text-slate-500 mt-1">Single short-answer evaluation with custom criterion weights.</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

                {/* Left: Form + Weights */}
                <div className="lg:col-span-7 space-y-5">

                  {/* Input form */}
                  <Card>
                    <CardHeader icon={AlignLeft} title="Grader Inputs" />
                    <form onSubmit={handleGradeSingle} className="p-5 space-y-4">
                      {[
                        { label: 'Context', value: singleContext,   set: setSingleContext,   req: false, placeholder: 'Background context (optional)…', rows: 1 },
                        { label: 'Question', value: singleQuestion, set: setSingleQuestion,  req: true,  placeholder: 'e.g. Explain photosynthesis.',   rows: 1 },
                        { label: 'Reference Answer', value: singleReference, set: setSingleReference, req: true, placeholder: 'Golden standard answer…', rows: 2 },
                        { label: 'Student Answer',   value: singleStudent,   set: setSingleStudent,   req: true, placeholder: 'Student response to grade…', rows: 3 },
                      ].map(({ label, value, set, req, placeholder, rows }) => (
                        <div key={label} className="space-y-1.5">
                          <label className="text-xs font-semibold text-slate-600">
                            {label}{req && <span className="text-rose-500 ml-0.5">*</span>}
                          </label>
                          {rows === 1 ? (
                            <input
                              type="text" value={value} required={req} placeholder={placeholder}
                              onChange={(e) => set(e.target.value)}
                              className="w-full text-sm bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-50 transition-all"
                            />
                          ) : (
                            <textarea
                              rows={rows} value={value} required={req} placeholder={placeholder}
                              onChange={(e) => set(e.target.value)}
                              className="w-full text-sm bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-50 transition-all resize-none"
                            />
                          )}
                        </div>
                      ))}

                      {singleError && (
                        <div className="flex items-start gap-2.5 p-3.5 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700 animate-fade-in">
                          <XCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" strokeWidth={2.5} />
                          {singleError}
                        </div>
                      )}

                      <button
                        type="submit" disabled={isGradingSingle}
                        className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-colors duration-150 shadow-sm cursor-pointer"
                      >
                        {isGradingSingle ? (
                          <><RefreshCw className="w-4 h-4 animate-spin" strokeWidth={2.5} /> Evaluating…</>
                        ) : (
                          <><Play className="w-4 h-4" strokeWidth={2.5} /> Run Evaluator</>
                        )}
                      </button>
                    </form>
                  </Card>

                  {/* Weights */}
                  <WeightsPanel
                    weights={weights} activePreset={activePreset}
                    onPresetChange={handlePresetChange} onWeightChange={handleWeightChange}
                  />
                </div>

                {/* Right: Results */}
                <div className="lg:col-span-5">
                  {singleResult ? (
                    <ResultPanel result={singleResult} />
                  ) : (
                    <Card>
                      <EmptyState
                        icon={Sliders}
                        title="Ready for Evaluation"
                        body="Submit a student answer on the left to view detailed model scores, tags, and AI feedback."
                      />
                    </Card>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* ── TAB: BATCH GRADER ───────────────────────────────────────── */}
          {activeTab === 'batch' && (
            <div className="max-w-6xl mx-auto animate-fade-up">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-xl font-semibold text-slate-800 tracking-tight">Batch Evaluation</h2>
                  <p className="text-sm text-slate-500 mt-1">Grade multiple student responses at once.</p>
                </div>
                <div className="flex gap-2">
                  {(['photosynthesis', 'watercycle'] as const).map(t => (
                    <button
                      key={t}
                      onClick={() => handleLoadSample(t)}
                      className="text-xs font-semibold text-slate-600 hover:text-indigo-700 bg-white hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 px-3.5 py-2 rounded-xl transition-all shadow-sm cursor-pointer"
                    >
                      {t === 'photosynthesis' ? '🌿 Photosynthesis' : '💧 Water Cycle'}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

                {/* Input panel */}
                <div className="lg:col-span-5">
                  <Card>
                    <CardHeader icon={Edit3} title="Batch Configuration" />
                    <div className="p-5 space-y-4">
                      {[
                        { label: 'Question Title', value: batchQuestion, set: setBatchQuestion, rows: 1, placeholder: 'e.g. Explain photosynthesis.' },
                        { label: 'Reference Answer', value: batchReference, set: setBatchReference, rows: 2, placeholder: 'Golden standard answer…' },
                      ].map(({ label, value, set, rows, placeholder }) => (
                        <div key={label} className="space-y-1.5">
                          <label className="text-xs font-semibold text-slate-600">{label}</label>
                          {rows === 1 ? (
                            <input type="text" value={value} placeholder={placeholder} onChange={(e) => set(e.target.value)}
                              className="w-full text-sm bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-50 transition-all" />
                          ) : (
                            <textarea rows={rows} value={value} placeholder={placeholder} onChange={(e) => set(e.target.value)}
                              className="w-full text-sm bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-50 transition-all resize-none" />
                          )}
                        </div>
                      ))}

                      <div className="space-y-1.5">
                        <label className="text-xs font-semibold text-slate-600">
                          Student Answers <span className="text-slate-400 font-normal">(format: Name: answer, one per line)</span>
                        </label>
                        <textarea rows={7} value={batchRawInput} onChange={(e) => setBatchRawInput(e.target.value)}
                          placeholder="Student A: Plants absorb sunlight…"
                          className="w-full text-xs bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-slate-700 placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-50 transition-all resize-none font-mono leading-relaxed" />
                      </div>

                      <button
                        onClick={handleGradeBatch}
                        disabled={isGradingBatch || !batchRawInput.trim()}
                        className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-sm"
                      >
                        {isGradingBatch ? (
                          <><RefreshCw className="w-4 h-4 animate-spin" strokeWidth={2.5} /> Grading in progress…</>
                        ) : (
                          <><Play className="w-4 h-4" strokeWidth={2.5} /> Evaluate Batch</>
                        )}
                      </button>
                    </div>
                  </Card>
                </div>

                {/* Results */}
                <div className="lg:col-span-7 space-y-5">
                  {batchItems.length > 0 ? (
                    <div className="animate-fade-up space-y-5">

                      {/* Table header controls */}
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-500">{batchItems.length} records</span>
                        <button
                          onClick={handleExportCSV}
                          className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 hover:text-slate-900 bg-white hover:bg-slate-50 border border-slate-200 px-3.5 py-2 rounded-xl transition-all shadow-sm cursor-pointer"
                        >
                          <Download className="w-3.5 h-3.5" strokeWidth={1.5} />
                          Export CSV
                        </button>
                      </div>

                      {/* Table */}
                      <Card className="overflow-hidden">
                        <table className="w-full text-left text-xs">
                          <thead>
                            <tr className="bg-slate-50 border-b border-slate-200">
                              {['Student', 'Answer', 'Grade', 'Status', ''].map(h => (
                                <th key={h} className="px-4 py-3 font-semibold text-slate-500 whitespace-nowrap">{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {batchItems.map((item) => {
                              const grade = item.result?.metrics.final_grade;
                              const tag   = item.result?.feedback.tags[0] || '';
                              const rowBg = item.status === 'success' && grade !== undefined
                                ? grade >= 80 ? 'bg-emerald-50/50 hover:bg-emerald-50'
                                : grade >= 55 ? 'bg-amber-50/50 hover:bg-amber-50'
                                : 'bg-rose-50/50 hover:bg-rose-50'
                                : 'bg-white hover:bg-slate-50';
                              const isSelected = selectedBatchItem?.id === item.id;
                              return (
                                <tr key={item.id}
                                  className={`border-b border-slate-100 transition-colors cursor-pointer ${rowBg} ${isSelected ? 'ring-2 ring-inset ring-indigo-300' : ''}`}
                                  onClick={() => item.result && setSelectedBatchItem(item)}
                                >
                                  <td className="px-4 py-3 font-semibold text-slate-800 whitespace-nowrap">{item.id}</td>
                                  <td className="px-4 py-3 text-slate-500 max-w-[160px] truncate">{item.student}</td>
                                  <td className="px-4 py-3 font-semibold tabular-nums text-slate-800 text-right whitespace-nowrap">
                                    {item.status === 'grading'
                                      ? <RefreshCw className="w-3.5 h-3.5 animate-spin ml-auto text-slate-400" />
                                      : item.status === 'success' ? `${grade}%`
                                      : <span className="text-rose-500">Err</span>}
                                  </td>
                                  <td className="px-4 py-3">
                                    {item.status === 'success' && tag && <Badge tag={tag} />}
                                  </td>
                                  <td className="px-4 py-3">
                                    {item.result && (
                                      <ChevronRight className={`w-3.5 h-3.5 ml-auto transition-colors ${isSelected ? 'text-indigo-500' : 'text-slate-300'}`} strokeWidth={2.5} />
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </Card>

                      {/* Detail panel */}
                      {selectedBatchItem?.result && (
                        <div className="animate-fade-up">
                          <Card>
                            <CardHeader
                              icon={Sparkles}
                              title={`Detail: ${selectedBatchItem.id}`}
                              right={
                                <div className="flex items-center gap-2">
                                  <Badge tag={selectedBatchItem.result.feedback.tags[0]} />
                                  <span className="text-sm font-semibold text-slate-800 tabular-nums">
                                    {selectedBatchItem.result.metrics.final_grade}%
                                  </span>
                                </div>
                              }
                            />
                            <div className="p-5 space-y-5">
                              <div className="grid grid-cols-2 gap-4 text-xs">
                                <div className="space-y-1.5">
                                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Explanation</p>
                                  <p className="text-slate-700 leading-relaxed">{selectedBatchItem.result.feedback.explanation}</p>
                                </div>
                                <div className="space-y-1.5">
                                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Recommendation</p>
                                  <p className="text-slate-700 leading-relaxed">{selectedBatchItem.result.feedback.suggestion}</p>
                                </div>
                              </div>

                              <div className="grid grid-cols-5 gap-2 pt-4 border-t border-slate-100">
                                {[
                                  { label: 'Semantic',  value: selectedBatchItem.result.metrics.semantic_score,  color: 'text-indigo-700 bg-indigo-50 border-indigo-200' },
                                  { label: 'Coverage',  value: selectedBatchItem.result.metrics.coverage_score,   color: 'text-sky-700 bg-sky-50 border-sky-200' },
                                  { label: 'Formality', value: selectedBatchItem.result.metrics.formality_score, color: 'text-slate-700 bg-slate-50 border-slate-200' },
                                  { label: 'Grammar',   value: selectedBatchItem.result.metrics.grammar_score,   color: 'text-orange-700 bg-orange-50 border-orange-200' },
                                  { label: 'Logic',     value: selectedBatchItem.result.metrics.logic_score,     color: 'text-purple-700 bg-purple-50 border-purple-200' },
                                ].map(({ label, value, color }) => (
                                  <div key={label} className={`p-3 rounded-xl border text-center ${color}`}>
                                    <p className="text-[9px] font-semibold uppercase tracking-wide">{label}</p>
                                    <p className="text-base font-semibold tabular-nums mt-0.5">{Math.round(value * 100)}%</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </Card>
                        </div>
                      )}
                    </div>
                  ) : (
                    <Card>
                      <EmptyState
                        icon={Layers}
                        title="No batch results yet"
                        body="Load a sample template or enter student answers on the left, then click Evaluate Batch."
                      />
                    </Card>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ── TAB: SETTINGS ───────────────────────────────────────────── */}
          {activeTab === 'settings' && (
            <div className="max-w-3xl mx-auto animate-fade-up space-y-6">
              <div className="mb-6">
                <h2 className="text-xl font-semibold text-slate-800 tracking-tight">System Configuration</h2>
                <p className="text-sm text-slate-500 mt-1">Configure backend endpoints and review scoring thresholds.</p>
              </div>

              {/* API URL */}
              <Card>
                <CardHeader icon={Database} title="FastAPI Connection" />
                <div className="p-5">
                  <div className="flex gap-3 items-end">
                    <div className="flex-1 space-y-1.5">
                      <label className="text-xs font-semibold text-slate-600">Server Endpoint URL</label>
                      <input
                        type="text" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)}
                        className="w-full text-xs bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-50 font-mono transition-all"
                      />
                    </div>
                    <button
                      onClick={checkHealth}
                      className="flex items-center gap-1.5 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 px-4 py-2.5 rounded-xl transition-all shadow-sm cursor-pointer whitespace-nowrap"
                    >
                      <RefreshCw className="w-3.5 h-3.5" strokeWidth={2} />
                      Test Connection
                    </button>
                  </div>
                </div>
              </Card>

              {/* Weights */}
              <WeightsPanel
                weights={weights} activePreset={activePreset}
                onPresetChange={handlePresetChange} onWeightChange={handleWeightChange}
              />

              {/* Thresholds */}
              <Card>
                <CardHeader icon={Shield} title="Active Grading Thresholds" right={<span className="text-[10px] text-slate-400 font-semibold">config.py</span>} />
                <div className="p-5">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
                    {[
                      {
                        title: 'Semantic Classification',
                        rows: [
                          { label: 'Correct threshold',    val: `≥ ${thresholds.semantic_correct}` },
                          { label: 'Partially correct',    val: `≥ ${thresholds.semantic_partial}` },
                          { label: 'Off-topic cap',        val: `< ${thresholds.semantic_off_topic}` },
                        ],
                      },
                      {
                        title: 'Keyword Coverage',
                        rows: [
                          { label: 'Correct coverage',    val: `≥ ${thresholds.coverage_correct}` },
                          { label: 'Good coverage',       val: `≥ ${thresholds.coverage_good}` },
                          { label: 'Missing concepts',    val: `< ${thresholds.coverage_missing}` },
                        ],
                      },
                      {
                        title: 'Contradiction NLI',
                        rows: [
                          { label: 'High (cap 40)',       val: `≥ ${thresholds.contradiction_high}` },
                          { label: 'Moderate (penalty)',  val: `≥ ${thresholds.contradiction_moderate}` },
                          { label: 'Grammar accept.',     val: `≥ ${thresholds.grammar_good}` },
                        ],
                      },
                    ].map(({ title, rows }) => (
                      <div key={title} className="space-y-2.5">
                        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">{title}</p>
                        {rows.map(({ label, val }) => (
                          <div key={label} className="flex justify-between border-b border-slate-100 pb-1.5">
                            <span className="text-slate-600">{label}</span>
                            <span className="font-semibold tabular-nums text-slate-800 font-mono">{val}</span>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              </Card>

            </div>
          )}

        </main>
      </div>
    </div>
  );
}
