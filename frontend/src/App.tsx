import React, { useState, useEffect } from 'react';
import {
  Play,
  Layers,
  Settings,
  XCircle,
  Sliders,
  Download,
  Shield,
  Database,
  ChevronRight,
  CheckCircle2,
  AlertOctagon,
  HelpCircle,
  Info,
  Sparkles,
  FlaskConical,
  Table2,
  LayoutDashboard,
  Wand2,
  Copy,
  Check,
  FileSpreadsheet,
  Radio,
  ExternalLink,
  Plus,
  RotateCcw,
  FileDown,
  ArrowRight,
  Search,
} from 'lucide-react';

import type {
  GradingResult,
  Weights,
  BatchItem,
  SystemThresholds,
  TabId,
  HealthStatus,
  ResearchConfig,
  AblationRow,
  SessionStats,
} from './types';

import {
  DEFAULT_WEIGHTS,
  WEIGHT_PRESETS,
  DEFAULT_THRESHOLDS,
  simulateGrading,
  calculateLocalGrade,
} from './mockData';

// shadcn/ui components
import { Button } from '@/components/ui/button';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Spinner } from '@/components/ui/spinner';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
import {
  Empty,
  EmptyHeader,
  EmptyTitle,
  EmptyDescription,
  EmptyMedia,
} from '@/components/ui/empty';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// ─── Score Helpers ────────────────────────────────────────────────────────────

const getScoreBadgeVariant = (score: number): { label: string; className: string } => {
  if (score >= 80) {
    return {
      label: 'Excellent',
      className: 'bg-emerald-50 text-emerald-700 border-emerald-200 border',
    };
  }
  if (score >= 55) {
    return {
      label: 'Partially Correct',
      className: 'bg-amber-50 text-amber-700 border-amber-200 border',
    };
  }
  return {
    label: 'Needs Work',
    className: 'bg-rose-50 text-rose-700 border-rose-200 border',
  };
};

const getTagBadgeStyle = (tag: string): string => {
  switch (tag) {
    case 'Correct':
      return 'bg-emerald-50 text-emerald-700 border-emerald-200 border';
    case 'Partially Correct':
      return 'bg-amber-50 text-amber-700 border-amber-200 border';
    case 'Incorrect':
      return 'bg-rose-50 text-rose-700 border-rose-200 border';
    case 'Missing Concepts':
      return 'bg-sky-50 text-sky-700 border-sky-200 border';
    case 'Factual Error':
    case 'Logical Error':
      return 'bg-red-50 text-red-700 border-red-200 border';
    case 'Grammar Error':
      return 'bg-orange-50 text-orange-700 border-orange-200 border';
    case 'Off-Topic':
      return 'bg-purple-50 text-purple-700 border-purple-200 border';
    default:
      return 'bg-[#F7F7F8] text-[#71717A] border-[#E5E7EB] border';
  }
};

const getTagIcon = (tag: string) => {
  switch (tag) {
    case 'Correct':
      return <CheckCircle2 className="size-3 text-emerald-700 shrink-0" strokeWidth={2.5} />;
    case 'Partially Correct':
      return <Info className="size-3 text-amber-700 shrink-0" strokeWidth={2.5} />;
    case 'Incorrect':
      return <XCircle className="size-3 text-rose-700 shrink-0" strokeWidth={2.5} />;
    case 'Factual Error':
    case 'Logical Error':
      return <AlertOctagon className="size-3 text-red-700 shrink-0" strokeWidth={2.5} />;
    default:
      return <HelpCircle className="size-3 text-[#71717A] shrink-0" strokeWidth={2} />;
  }
};

// ─── Circular Score Ring ──────────────────────────────────────────────────────

function ScoreGauge({ score, size = 110 }: { score: number; size?: number }) {
  const circ = 283;
  const offset = circ - (circ * Math.min(100, Math.max(0, score))) / 100;
  const isHigh = score >= 80;
  const isMid = score >= 55 && score < 80;
  const gradId = isHigh ? 'scoreGradHigh' : isMid ? 'scoreGradMid' : 'scoreGradLow';
  const glowColor = isHigh
    ? 'rgba(16, 185, 129, 0.15)'
    : isMid
    ? 'rgba(245, 158, 11, 0.15)'
    : 'rgba(239, 68, 68, 0.15)';

  return (
    <div
      className="relative shrink-0 rounded-full flex items-center justify-center p-1 transition-all duration-500"
      style={{
        width: size,
        height: size,
        boxShadow: `0 8px 24px -4px ${glowColor}`,
      }}
    >
      <svg className="size-full" viewBox="0 0 36 36">
        <defs>
          <linearGradient id="scoreGradHigh" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#34D399" />
            <stop offset="100%" stopColor="#059669" />
          </linearGradient>
          <linearGradient id="scoreGradMid" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#FBBF24" />
            <stop offset="100%" stopColor="#D97706" />
          </linearGradient>
          <linearGradient id="scoreGradLow" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#FB7185" />
            <stop offset="100%" stopColor="#E11D48" />
          </linearGradient>
        </defs>
        <path
          strokeWidth="2.5"
          fill="none"
          stroke="#F4F4F5"
          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
        />
        <path
          className="progress-circle"
          strokeWidth="3"
          strokeLinecap="round"
          fill="none"
          stroke={`url(#${gradId})`}
          style={{ '--offset': offset } as React.CSSProperties}
          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-semibold tabular-nums text-[#18181B] leading-none">
          {Math.round(score)}
        </span>
        <span className="text-[10px] font-normal text-[#71717A] leading-none mt-1">/ 100</span>
      </div>
    </div>
  );
}

// ─── Feature Breakdown Progress Bar ───────────────────────────────────────────

function FeatureBar({
  label,
  value,
  colorHex,
  weightPct,
  warningBadge,
}: {
  label: string;
  value: number;
  colorHex: string;
  weightPct?: number;
  warningBadge?: string;
}) {
  const pct = Math.round(Math.min(100, Math.max(0, value * 100)));
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-center text-xs">
        <div className="flex items-center gap-1.5">
          <span className="font-normal text-[#71717A]">{label}</span>
          {weightPct !== undefined && (
            <span className="text-[10px] font-normal text-[#71717A] bg-[#F7F7F8] px-1.5 py-0.2 rounded border border-[#E5E7EB]">
              {weightPct}% wt
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {warningBadge && (
            <span className="text-[10px] font-semibold text-rose-600 bg-rose-50 px-2 py-0.5 rounded-full border border-rose-200">
              {warningBadge}
            </span>
          )}
          <span className="font-semibold text-[#18181B] tabular-nums">{pct}%</span>
        </div>
      </div>
      <div className="h-2 w-full rounded-full bg-[#F7F7F8] overflow-hidden border border-[#E5E7EB]">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: colorHex }}
        />
      </div>
    </div>
  );
}

// ─── Main Application ─────────────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('dashboard');
  const [isLive, setIsLive] = useState<boolean>(false);
  const [apiUrl, setApiUrl] = useState<string>('http://127.0.0.1:8000');
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>({
    status: 'healthy',
    device: 'Client Sim',
    models_loaded: true,
  });

  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const [activePreset, setActivePreset] = useState<string>('balanced');
  const [thresholds] = useState<SystemThresholds>(DEFAULT_THRESHOLDS);

  // Single Grader state
  const [singleContext, setSingleContext] = useState<string>(
    'Photosynthesis is the process by which green plants and some organisms use sunlight to synthesize nutrients from carbon dioxide and water.'
  );
  const [singleQuestion, setSingleQuestion] = useState<string>(
    'Explain the process of photosynthesis and its primary biological importance.'
  );
  const [singleReference, setSingleReference] = useState<string>(
    'Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll. It provides chemical energy for plants and oxygen for aerobic respiration.'
  );
  const [singleStudent, setSingleStudent] = useState<string>(
    'Plants use sunlight, CO2, and water to make glucose and oxygen through chlorophyll. This process provides food and releases oxygen into the atmosphere.'
  );
  const [singleResult, setSingleResult] = useState<GradingResult | null>(null);
  const [isGradingSingle, setIsGradingSingle] = useState<boolean>(false);
  const [singleError, setSingleError] = useState<string | null>(null);
  const [activeBenchmark, setActiveBenchmark] = useState<string>('photosynthesis');
  const [copiedFeedback, setCopiedFeedback] = useState<boolean>(false);

  // Batch Grader state
  const [batchQuestion, setBatchQuestion] = useState<string>(
    'Explain the process of photosynthesis and its primary biological importance.'
  );
  const [batchReference, setBatchReference] = useState<string>(
    'Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll. It produces food and oxygen.'
  );
  const [batchRawInput, setBatchRawInput] = useState<string>(
    `Alice: Plants use sunlight, carbon dioxide, and water to make glucose and release oxygen via chlorophyll.\nBob: Plants require sunlight and water to produce food. They release oxygen gas into the air.\nCharlie: Animals eat plants to absorb solar energy directly from chlorophyll pigments.\nDavid: Sunlight and water create green leaves.`
  );
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);
  const [isGradingBatch, setIsGradingBatch] = useState<boolean>(false);
  const [selectedBatchItem, setSelectedBatchItem] = useState<BatchItem | null>(null);
  const [batchFilter, setBatchFilter] = useState<'all' | 'correct' | 'partial' | 'incorrect'>('all');
  const [batchSearch, setBatchSearch] = useState<string>('');

  // Research state
  const [researchConfig, setResearchConfig] = useState<ResearchConfig | null>(null);
  const [ablationResults, setAblationResults] = useState<AblationRow[]>([]);
  const [ablationLatex, setAblationLatex] = useState<string>('');
  const [isRunningAblation, setIsRunningAblation] = useState<boolean>(false);
  const [ablationCachePath, setAblationCachePath] = useState<string>('./cache/features.pkl');
  const [ablationError, setAblationError] = useState<string | null>(null);
  const [ablationSeed, setAblationSeed] = useState<number>(42);
  const [copiedLatex, setCopiedLatex] = useState<boolean>(false);

  // Stats
  const [stats, setStats] = useState<SessionStats>({
    totalGraded: 4,
    averageGrade: 78.5,
    correctCount: 2,
    partialCount: 1,
    incorrectCount: 1,
  });

  // Health check
  const checkHealth = async () => {
    try {
      const res = await fetch(`${apiUrl}/health`);
      if (res.ok) {
        const data: HealthStatus = await res.json();
        setHealthStatus(data);
      } else {
        setHealthStatus({ status: 'offline', device: 'None', models_loaded: false });
      }
    } catch {
      setHealthStatus({ status: 'offline', device: 'None', models_loaded: false });
    }
  };

  useEffect(() => {
    let active = true;
    if (isLive) {
      fetch(`${apiUrl}/health`)
        .then((res) => (res.ok ? res.json() : { status: 'offline', device: 'None', models_loaded: false }))
        .then((data: HealthStatus) => {
          if (active) setHealthStatus(data);
        })
        .catch(() => {
          if (active) setHealthStatus({ status: 'offline', device: 'None', models_loaded: false });
        });
    }
    return () => {
      active = false;
    };
  }, [isLive, apiUrl]);

  useEffect(() => {
    let active = true;
    if (activeTab === 'research' && isLive) {
      fetch(`${apiUrl}/research/config`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (active && data) setResearchConfig(data);
        })
        .catch(() => {
          if (active) setResearchConfig(null);
        });
    }
    return () => {
      active = false;
    };
  }, [activeTab, isLive, apiUrl]);

  // Recalculate helper
  const recalculateSingle = async (nw: Weights) => {
    if (!singleResult) return;
    if (isLive) {
      try {
        const res = await fetch(`${apiUrl}/grade/recalculate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ metrics: singleResult.metrics, weights: nw }),
        });
        if (res.ok) {
          const data = await res.json();
          setSingleResult((prev) =>
            prev ? { ...prev, metrics: { ...prev.metrics, final_grade: data.new_grade } } : null
          );
        }
      } catch {
        const g = calculateLocalGrade(singleResult.metrics, nw);
        setSingleResult((prev) =>
          prev ? { ...prev, metrics: { ...prev.metrics, final_grade: g } } : null
        );
      }
    } else {
      const g = calculateLocalGrade(singleResult.metrics, nw);
      setSingleResult((prev) =>
        prev ? { ...prev, metrics: { ...prev.metrics, final_grade: g } } : null
      );
    }
  };

  const handlePresetChange = (presetName: string) => {
    setActivePreset(presetName);
    if (WEIGHT_PRESETS[presetName]) {
      const nw = WEIGHT_PRESETS[presetName];
      setWeights(nw);
      if (singleResult) recalculateSingle(nw);
    }
  };

  const handleWeightChange = (key: keyof Weights, value: number) => {
    setActivePreset('custom');
    const nw = { ...weights, [key]: Math.round(value * 100) / 100 };
    setWeights(nw);
    if (singleResult) recalculateSingle(nw);
  };

  const updateStats = (score: number, tags: string[]) => {
    setStats((prev) => {
      const nt = prev.totalGraded + 1;
      return {
        totalGraded: nt,
        averageGrade:
          Math.round(((prev.averageGrade * prev.totalGraded + score) / nt) * 100) / 100,
        correctCount: prev.correctCount + (tags.includes('Correct') ? 1 : 0),
        partialCount: prev.partialCount + (tags.includes('Partially Correct') ? 1 : 0),
        incorrectCount:
          prev.incorrectCount +
          (!tags.includes('Correct') && !tags.includes('Partially Correct') ? 1 : 0),
      };
    });
  };

  const handleGradeSingle = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!singleQuestion.trim() || !singleReference.trim() || !singleStudent.trim()) return;

    setIsGradingSingle(true);
    setSingleError(null);

    if (isLive) {
      try {
        const res = await fetch(`${apiUrl}/grade`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            context: singleContext,
            question: singleQuestion,
            reference: singleReference,
            student: singleStudent,
            weights,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.success) {
            setSingleResult(data.data);
            updateStats(data.data.metrics.final_grade, data.data.feedback.tags);
          } else {
            setSingleError(data.error || 'Evaluation failed on server.');
          }
        } else {
          const errData = await res.json().catch(() => ({}));
          setSingleError(errData.detail || `Server error (${res.status})`);
        }
      } catch {
        setSingleError('Cannot connect to Live API. Check if uvicorn is running on port 8000.');
      } finally {
        setIsGradingSingle(false);
      }
    } else {
      setTimeout(() => {
        try {
          const r = simulateGrading(
            singleContext,
            singleQuestion,
            singleReference,
            singleStudent,
            weights
          );
          setSingleResult(r);
          updateStats(r.metrics.final_grade, r.feedback.tags);
        } catch {
          setSingleError('Simulation evaluation failed.');
        } finally {
          setIsGradingSingle(false);
        }
      }, 650);
    }
  };

  const handleGradeBatch = async () => {
    if (!batchRawInput.trim()) return;
    setIsGradingBatch(true);
    setSelectedBatchItem(null);

    const lines = batchRawInput.split('\n').filter((l) => l.trim());
    const items: BatchItem[] = lines.map((line, idx) => {
      let id = `Student ${idx + 1}`;
      let student = line;
      const ci = line.indexOf(':');
      if (ci > 0 && ci < 30) {
        id = line.substring(0, ci).trim();
        student = line.substring(ci + 1).trim();
      }
      return {
        id,
        context: '',
        question: batchQuestion,
        reference: batchReference,
        student,
        status: 'grading',
      };
    });
    setBatchItems(items);

    if (isLive) {
      try {
        const res = await fetch(`${apiUrl}/grade/batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            items: items.map((i) => ({
              context: i.context,
              question: i.question,
              reference: i.reference,
              student: i.student,
            })),
            weights,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.success && data.data.results) {
            setBatchItems((prev) =>
              prev.map((item, idx) => ({
                ...item,
                status: 'success',
                result: data.data.results[idx],
              }))
            );
          } else {
            throw new Error(data.message || 'Batch evaluation error');
          }
        } else {
          throw new Error(`HTTP ${res.status}`);
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Batch grading error';
        setBatchItems((prev) => prev.map((item) => ({ ...item, status: 'failed', error: msg })));
      } finally {
        setIsGradingBatch(false);
      }
    } else {
      for (let i = 0; i < items.length; i++) {
        await new Promise((r) => setTimeout(r, 280));
        const res = simulateGrading('', items[i].question, items[i].reference, items[i].student, weights);
        setBatchItems((prev) => {
          const u = [...prev];
          u[i] = { ...u[i], status: 'success', result: res };
          return u;
        });
      }
      setIsGradingBatch(false);
    }
  };

  const [pingLatency, setPingLatency] = useState<number | null>(null);

  const testPing = async () => {
    const t0 = performance.now();
    await checkHealth();
    const t1 = performance.now();
    setPingLatency(Math.round(t1 - t0));
  };

  const handleClearWorkspace = () => {
    setSingleContext('');
    setSingleQuestion('');
    setSingleReference('');
    setSingleStudent('');
    setSingleResult(null);
    setSingleError(null);
    setActiveBenchmark('custom');
  };

  const handleCopyFeedback = () => {
    if (!singleResult) return;
    const content = `[AutoGrade Diagnostic Feedback]\nScore: ${Math.round(singleResult.metrics.final_grade)}/100\nTags: ${singleResult.feedback.tags.join(', ')}\n\nExplanation:\n${singleResult.feedback.explanation}\n\nRecommendation:\n${singleResult.feedback.suggestion}`;
    navigator.clipboard.writeText(content);
    setCopiedFeedback(true);
    setTimeout(() => setCopiedFeedback(false), 2000);
  };

  const handleInspectBatchInWorkspace = (item: BatchItem) => {
    setSingleQuestion(item.question);
    setSingleReference(item.reference);
    setSingleStudent(item.student);
    setSingleContext(item.context || '');
    if (item.result) {
      setSingleResult(item.result);
    } else {
      setSingleResult(null);
    }
    setActiveBenchmark('custom');
    setActiveTab('single');
  };

  const handleNormalizeWeights = () => {
    setActivePreset('balanced');
    setWeights(DEFAULT_WEIGHTS);
    if (singleResult) recalculateSingle(DEFAULT_WEIGHTS);
  };

  const downloadLatex = () => {
    if (!ablationLatex) return;
    const blob = new Blob([ablationLatex], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'ablation_study_table.tex';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const downloadAblationJson = () => {
    if (!ablationResults.length) return;
    const blob = new Blob([JSON.stringify(ablationResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'ablation_study_results.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleLoadSample = (
    topic: 'photosynthesis' | 'watercycle' | 'contradiction' | 'informal'
  ) => {
    setActiveBenchmark(topic);
    if (topic === 'photosynthesis') {
      setSingleQuestion('Explain the process of photosynthesis and its primary biological importance.');
      setSingleReference(
        'Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll. It provides chemical energy for plants and oxygen for aerobic respiration.'
      );
      setSingleStudent(
        'Plants use sunlight, CO2, and water to make glucose and oxygen through chlorophyll. This process provides food and releases oxygen into the atmosphere.'
      );
      setBatchQuestion('Explain the process of photosynthesis.');
      setBatchReference(
        'Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.'
      );
      setBatchRawInput(
        `Alice: Plants use sunlight, carbon dioxide, and water to make glucose and release oxygen via chlorophyll.\nBob: Plants require sunlight and water to produce food. They release oxygen gas.\nCharlie: Animals eat plants to absorb chemical energy created by chloroplasts.\nDavid: Green plants grow with sunlight.`
      );
    } else if (topic === 'watercycle') {
      setSingleQuestion('Describe the water cycle and explain how solar heat drives it.');
      setSingleReference(
        'The water cycle describes the continuous movement of water on, above, and below Earth surface. Solar radiation evaporates water from oceans, creating vapor that condenses into clouds and falls as precipitation.'
      );
      setSingleStudent(
        'Water evaporates from oceans due to solar heat, condenses to form clouds in the sky, and then precipitates back to the surface as rain.'
      );
      setBatchQuestion('Describe the water cycle.');
      setBatchReference(
        'The water cycle is the continuous movement where water evaporates, condenses into clouds, and falls as precipitation.'
      );
      setBatchRawInput(
        `John: Water evaporates from oceans, condenses into clouds, and precipitates back to Earth.\nSarah: Evaporation, condensation, and precipitation drive continuous hydrological movement.\nKevin: Water heats up and goes up then rains many times.\nEmma: Rain comes from clouds in the sky.`
      );
    } else if (topic === 'contradiction') {
      setSingleQuestion('Explain the process of photosynthesis and its primary biological importance.');
      setSingleReference(
        'Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll. It provides chemical energy for plants and oxygen for aerobic respiration.'
      );
      setSingleStudent(
        'Animals eat plants to perform photosynthesis and destroy oxygen gas instead of releasing it into the atmosphere.'
      );
    } else if (topic === 'informal') {
      setSingleQuestion('Explain the process of photosynthesis and its primary biological importance.');
      setSingleReference(
        'Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll. It provides chemical energy for plants and oxygen for aerobic respiration.'
      );
      setSingleStudent(
        'plants do stuff with the sun and water to make food bro, it gives energy to things and smells like leaves.'
      );
    }
  };

  const handleExportCSV = () => {
    if (!batchItems.length) return;
    let csv = 'data:text/csv;charset=utf-8,Student,Answer,Grade,Correctness,Explanation\n';
    batchItems.forEach((item) => {
      const ans = `"${item.student.replace(/"/g, '""')}"`;
      const grade = item.result?.metrics.final_grade ?? 'N/A';
      const tag = item.result?.feedback.tags[0] ?? 'N/A';
      const exp = item.result
        ? `"${item.result.feedback.explanation.replace(/"/g, '""')}"`
        : 'N/A';
      csv += `${item.id},${ans},${grade},${tag},${exp}\n`;
    });
    const link = document.createElement('a');
    link.setAttribute('href', encodeURI(csv));
    link.setAttribute('download', 'grading_results.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleRunAblation = async () => {
    setIsRunningAblation(true);
    setAblationError(null);
    setAblationResults([]);

    if (!isLive) {
      await new Promise((r) => setTimeout(r, 1000));
      setAblationResults([
        {
          Variant: 'Full Model (Proposed)',
          Accuracy: '0.742',
          'Macro F1': '0.718',
          QWK: '0.701',
          'Δ F1': '—',
          'Δ QWK': '—',
          'Sig.': '',
        },
        {
          Variant: 'FM-S (−Semantic)',
          Accuracy: '0.611',
          'Macro F1': '0.589',
          QWK: '0.562',
          'Δ F1': '-0.129',
          'Δ QWK': '-0.139',
          'Sig.': '***',
        },
        {
          Variant: 'FM-C (−Coverage)',
          Accuracy: '0.698',
          'Macro F1': '0.674',
          QWK: '0.651',
          'Δ F1': '-0.044',
          'Δ QWK': '-0.050',
          'Sig.': '*',
        },
        {
          Variant: 'FM-F (−Formality)',
          Accuracy: '0.751',
          'Macro F1': '0.726',
          QWK: '0.709',
          'Δ F1': '+0.008',
          'Δ QWK': '+0.008',
          'Sig.': 'ns',
        },
        {
          Variant: 'FM-G (−Grammar)',
          Accuracy: '0.729',
          'Macro F1': '0.703',
          QWK: '0.685',
          'Δ F1': '-0.015',
          'Δ QWK': '-0.016',
          'Sig.': 'ns',
        },
        {
          Variant: 'FM-L (−Logic/NLI)',
          Accuracy: '0.684',
          'Macro F1': '0.661',
          QWK: '0.637',
          'Δ F1': '-0.057',
          'Δ QWK': '-0.064',
          'Sig.': '**',
        },
        {
          Variant: 'FM-LLM (−Synthesis)',
          Accuracy: '0.719',
          'Macro F1': '0.695',
          QWK: '0.673',
          'Δ F1': '-0.023',
          'Δ QWK': '-0.028',
          'Sig.': '*',
        },
      ]);
      setAblationLatex(
        `% Ablation Study Results — AutoGradeSystem\n\\begin{table}[t]\n\\centering\n\\caption{Ablation Study Results with Wilcoxon Signed-Rank Tests}\n\\begin{tabular}{lcccccc}\n\\toprule\nModel Variant & Acc & F1 & QWK & $\\Delta$F1 & $\\Delta$QWK & Sig. \\\\\n\\midrule\nFull Model & 0.742 & 0.718 & 0.701 & --- & --- & --- \\\\\nFM-S & 0.611 & 0.589 & 0.562 & -0.129 & -0.139 & $^{***}$ \\\\\nFM-C & 0.698 & 0.674 & 0.651 & -0.044 & -0.050 & $^{*}$ \\\\\nFM-L & 0.684 & 0.661 & 0.637 & -0.057 & -0.064 & $^{**}$ \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}`
      );
      setIsRunningAblation(false);
      return;
    }

    try {
      const res = await fetch(`${apiUrl}/research/ablation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cache_path: ablationCachePath,
          random_seed: ablationSeed,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          setAblationResults(data.results_table);
          setAblationLatex(data.latex_snippet);
        } else {
          setAblationError(data.error || 'Ablation experiment failed.');
        }
      } else {
        const e = await res.json().catch(() => ({}));
        setAblationError(e.detail || `Server error (${res.status})`);
      }
    } catch {
      setAblationError('Failed to communicate with research backend endpoint.');
    } finally {
      setIsRunningAblation(false);
    }
  };

  const copyLatex = () => {
    if (!ablationLatex) return;
    navigator.clipboard.writeText(ablationLatex);
    setCopiedLatex(true);
    setTimeout(() => setCopiedLatex(false), 2000);
  };

  // ─── Sidebar Navigation Items ───────────────────────────────────────────────
  const navTabs: { id: TabId; label: string; icon: React.ElementType }[] = [
    { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
    { id: 'single', label: 'Single Grader', icon: Wand2 },
    { id: 'batch', label: 'Batch Grader', icon: Layers },
    { id: 'research', label: 'Research Lab', icon: FlaskConical },
    { id: 'settings', label: 'System Settings', icon: Settings },
  ];

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-[#F7F7F8] text-[#18181B] flex flex-row font-sans antialiased selection:bg-[#F04F9B]/20">
        
        {/* ══════════════════════════════════════════════════════════════════════
            SLIM VERTICAL SIDEBAR (Magnific Reference B)
        ══════════════════════════════════════════════════════════════════════ */}
        <aside className="w-16 md:w-20 bg-white border-r border-[#E5E7EB] flex flex-col items-center py-5 shrink-0 z-30 select-none">
          
          {/* Brand Monogram */}
          <div className="mb-6 flex flex-col items-center gap-1">
            <div className="size-10 rounded-2xl bg-[#181113] flex items-center justify-center shadow-sm relative group cursor-pointer">
              <span className="text-white font-semibold text-base tracking-tighter">AG</span>
              <span className="absolute -bottom-1 -right-1 size-3 rounded-full bg-[#F04F9B] border-2 border-white" />
            </div>
          </div>

          {/* Quick Action Pill Button (Magnific '+' Action) */}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => setActiveTab('single')}
                className="size-10 rounded-2xl bg-[#F04F9B] text-white flex items-center justify-center hover:opacity-90 transition-all shadow-sm mb-6 active:scale-95 cursor-pointer"
                aria-label="New Evaluation"
              >
                <Plus className="size-5" strokeWidth={2.5} />
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <span className="text-xs font-semibold">New Evaluation</span>
            </TooltipContent>
          </Tooltip>

          <Separator className="w-8 mb-4 bg-[#E5E7EB]" />

          {/* Nav Icons Stack */}
          <nav className="flex flex-col items-center gap-2 flex-1 w-full px-2">
            {navTabs.map(({ id, label, icon: Icon }) => {
              const active = activeTab === id;
              return (
                <Tooltip key={id}>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setActiveTab(id)}
                      className={`relative size-10 md:size-11 rounded-2xl flex items-center justify-center transition-all cursor-pointer ${
                        active
                          ? 'bg-[#18181B] text-white shadow-sm'
                          : 'text-[#71717A] hover:text-[#18181B] hover:bg-[#F7F7F8]'
                      }`}
                      aria-label={label}
                    >
                      <Icon
                        className="size-5"
                        strokeWidth={active ? 2.5 : 2}
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <span className="text-xs font-semibold">{label}</span>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </nav>

          {/* Bottom Diagnostics / Status */}
          <div className="mt-auto flex flex-col items-center gap-3">
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={`size-3 rounded-full border-2 border-white cursor-pointer ${
                    healthStatus?.status === 'healthy' ? 'bg-emerald-500' : 'bg-rose-500'
                  }`}
                />
              </TooltipTrigger>
              <TooltipContent side="right">
                <span className="text-xs font-normal">
                  Backend: {healthStatus?.status ?? 'Connecting...'} ({healthStatus?.device ?? 'None'})
                </span>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <a
                  href={`${apiUrl}/docs`}
                  target="_blank"
                  rel="noreferrer"
                  className="size-9 rounded-xl flex items-center justify-center text-[#71717A] hover:text-[#18181B] hover:bg-[#F7F7F8] transition-colors"
                  aria-label="API Swagger Docs"
                >
                  <ExternalLink className="size-4" strokeWidth={2} />
                </a>
              </TooltipTrigger>
              <TooltipContent side="right">
                <span className="text-xs font-normal">Open Swagger API Docs</span>
              </TooltipContent>
            </Tooltip>
          </div>
        </aside>

        {/* ══════════════════════════════════════════════════════════════════════
            MAIN CONTENT AREA
        ══════════════════════════════════════════════════════════════════════ */}
        <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
          
          {/* ── Compact Header (Magnific Reference B & C) ────────────────────── */}
          <header className="sticky top-0 z-20 bg-white/95 backdrop-blur-md border-b border-[#E5E7EB] px-6 py-3.5 flex items-center justify-between">
            
            {/* Breadcrumb Context */}
            <div className="flex items-center gap-2 text-xs">
              <span className="font-semibold text-[#18181B] tracking-tight">AutoGrade</span>
              <span className="text-[#71717A] font-normal">/</span>
              <span className="font-semibold text-[#71717A] capitalize">
                {activeTab === 'dashboard'
                  ? 'Overview & Architecture'
                  : activeTab === 'single'
                  ? 'Workspace · Single Evaluation'
                  : activeTab === 'batch'
                  ? 'Multi-Student Assessment'
                  : activeTab === 'research'
                  ? 'Research Lab · Ablation Study'
                  : 'System Settings'}
              </span>
            </div>

            {/* Header Right Controls */}
            <div className="flex items-center gap-3">
              
              {/* Live / Simulated Toggle */}
              <div className="flex items-center bg-[#F7F7F8] border border-[#E5E7EB] rounded-full p-0.5">
                <button
                  onClick={() => setIsLive(false)}
                  className={`px-3 py-1 text-xs font-semibold rounded-full transition-all cursor-pointer ${
                    !isLive
                      ? 'bg-white text-[#18181B] shadow-xs'
                      : 'text-[#71717A] hover:text-[#18181B]'
                  }`}
                >
                  Simulated
                </button>
                <button
                  onClick={() => setIsLive(true)}
                  className={`px-3 py-1 text-xs font-semibold rounded-full transition-all cursor-pointer ${
                    isLive
                      ? 'bg-[#18181B] text-white shadow-xs'
                      : 'text-[#71717A] hover:text-[#18181B]'
                  }`}
                >
                  Live API
                </button>
              </div>

              {/* Status Badge */}
              <Badge
                variant="outline"
                className={`text-xs px-2.5 py-1 font-semibold rounded-full ${
                  isLive
                    ? healthStatus?.status === 'healthy'
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-rose-50 text-rose-700 border-rose-200'
                    : 'bg-[#F7F7F8] text-[#71717A] border-[#E5E7EB]'
                }`}
              >
                <Radio className="size-3 shrink-0" strokeWidth={2.5} />
                <span>{isLive ? `Live · ${healthStatus?.device ?? 'CPU'}` : 'Simulated'}</span>
              </Badge>

              {/* Quick Sample Trigger */}
              <div className="hidden sm:flex items-center gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleLoadSample('photosynthesis')}
                  className="rounded-full text-xs font-semibold text-[#18181B] border-[#E5E7EB] hover:bg-[#F7F7F8]"
                >
                  Photosynthesis Sample
                </Button>
              </div>
            </div>
          </header>

          {/* ── Main Tab Contents ────────────────────────────────────────────── */}
          <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto pb-36">

            {/* ══════════════════════════════════════════════════════════════════
                TAB: DASHBOARD / OVERVIEW (Reference A & B)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'dashboard' && (
              <div className="flex flex-col gap-8 animate-fade-up">

                {/* ── Reference A: Premium Cinematic Presentation Hero ───────── */}
                <div className="relative rounded-3xl overflow-hidden bg-[#181113] text-white p-8 md:p-12 border border-[#2A2024] shadow-md">
                  {/* Atmospheric Glow */}
                  <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 rounded-full bg-[#F04F9B]/15 blur-3xl pointer-events-none" />
                  <div className="absolute bottom-0 left-1/3 -mb-20 w-80 h-80 rounded-full bg-[#5268F5]/10 blur-3xl pointer-events-none" />

                  <div className="relative z-10 max-w-2xl flex flex-col gap-4">
                    {/* Pill Tag */}
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md border border-white/15 w-fit">
                      <span className="size-2 rounded-full bg-[#F04F9B]" />
                      <span className="text-xs font-semibold text-white/90">
                        Hybrid ASAG Architecture · 5 Lightweight Models + Qwen2.5
                      </span>
                    </div>

                    {/* Headline */}
                    <h1 className="text-3xl md:text-5xl font-semibold tracking-tight text-white leading-tight">
                      The automated grading platform for high-precision short answers.
                    </h1>

                    {/* Subtitle */}
                    <p className="text-sm md:text-base font-normal text-white/70 leading-relaxed">
                      DeBERTa NLI logical verification, SimCSE semantic similarity, and KeyBERT coverage combined into transparent, explainable feedback.
                    </p>

                    {/* Action Buttons */}
                    <div className="flex flex-wrap items-center gap-3 pt-2">
                      <Button
                        onClick={() => setActiveTab('single')}
                        className="rounded-full bg-white text-[#181113] hover:bg-white/90 font-semibold px-6 py-2.5 h-auto text-sm shadow-sm"
                      >
                        <Play className="size-4 shrink-0 fill-current" strokeWidth={2.5} />
                        Start Evaluating
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => setActiveTab('research')}
                        className="rounded-full bg-transparent text-white border-white/20 hover:bg-white/10 font-semibold px-6 py-2.5 h-auto text-sm"
                      >
                        <FlaskConical className="size-4 shrink-0" strokeWidth={2.5} />
                        Explore Research Lab
                      </Button>
                    </div>
                  </div>

                  {/* Right side floating metrics badge */}
                  <div className="mt-8 pt-6 border-t border-white/10 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-normal text-white/60">
                    <div>
                      <p className="font-semibold text-white text-base">5 Models</p>
                      <p className="text-xs text-white/60">Parallel CPU Inference</p>
                    </div>
                    <div>
                      <p className="font-semibold text-white text-base">&lt; 50ms</p>
                      <p className="text-xs text-white/60">Average Latency</p>
                    </div>
                    <div>
                      <p className="font-semibold text-white text-base">0.742 Acc</p>
                      <p className="text-xs text-white/60">SemEval Benchmark</p>
                    </div>
                    <div>
                      <p className="font-semibold text-white text-base">4-bit NF4</p>
                      <p className="text-xs text-white/60">Quantized Synthesis</p>
                    </div>
                  </div>
                </div>

                {/* ── Reference B: Application Dashboard Cards Grid ─────────── */}
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-[#18181B] tracking-tight">
                      Evaluation Metrics
                    </h2>
                    <span className="text-xs font-normal text-[#71717A]">
                      Aggregated active session
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <Card className="rounded-2xl border-[#E5E7EB] bg-white shadow-xs">
                      <CardContent className="p-5 flex flex-col gap-1">
                        <span className="text-xs font-normal text-[#71717A]">Total Evaluated</span>
                        <span className="text-3xl font-semibold tabular-nums text-[#18181B]">
                          {stats.totalGraded}
                        </span>
                        <span className="text-[11px] font-normal text-[#71717A]">Active runs</span>
                      </CardContent>
                    </Card>

                    <Card className="rounded-2xl border-[#E5E7EB] bg-white shadow-xs">
                      <CardContent className="p-5 flex flex-col gap-1">
                        <span className="text-xs font-normal text-[#71717A]">Average Grade</span>
                        <span className="text-3xl font-semibold tabular-nums text-[#18181B]">
                          {stats.averageGrade}%
                        </span>
                        <Progress value={stats.averageGrade} className="h-1.5 mt-1" />
                      </CardContent>
                    </Card>

                    <Card className="rounded-2xl border-[#E5E7EB] bg-white shadow-xs">
                      <CardContent className="p-5 flex flex-col gap-1">
                        <span className="text-xs font-normal text-[#71717A]">Pass Rate</span>
                        <span className="text-3xl font-semibold tabular-nums text-emerald-600">
                          {Math.round((stats.correctCount / (stats.totalGraded || 1)) * 100)}%
                        </span>
                        <span className="text-[11px] font-normal text-[#71717A]">
                          {stats.correctCount} Correct tags
                        </span>
                      </CardContent>
                    </Card>

                    <Card className="rounded-2xl border-[#E5E7EB] bg-white shadow-xs">
                      <CardContent className="p-5 flex flex-col gap-1">
                        <span className="text-xs font-normal text-[#71717A]">Penalized / Capped</span>
                        <span className="text-3xl font-semibold tabular-nums text-rose-600">
                          {stats.incorrectCount}
                        </span>
                        <span className="text-[11px] font-normal text-[#71717A]">
                          Logic contradiction
                        </span>
                      </CardContent>
                    </Card>
                  </div>
                </div>

                {/* ── 5-Model Pipeline Showcase Bento ──────────────────────── */}
                <div className="flex flex-col gap-3">
                  <h2 className="text-lg font-semibold text-[#18181B] tracking-tight">
                    Pipeline Architecture
                  </h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[
                      {
                        step: '01',
                        name: 'Semantic Similarity',
                        model: 'princeton-nlp/sup-simcse-roberta-large',
                        desc: 'Sentence embedding cosine similarity with human grading alignment.',
                        badge: 'Embedding',
                        color: '#5268F5',
                      },
                      {
                        step: '02',
                        name: 'Keyword Coverage',
                        model: 'all-MiniLM-L6-v2 · KeyBERT',
                        desc: 'N-gram key concept extraction ensuring domain vocabulary is covered.',
                        badge: 'Vocabulary',
                        color: '#0EA5E9',
                      },
                      {
                        step: '03',
                        name: 'Academic Formality',
                        model: 'cointegrated/roberta-base-formality',
                        desc: 'Evaluates tone formality, rejecting casual chat or slang answers.',
                        badge: 'Style',
                        color: '#64748B',
                      },
                      {
                        step: '04',
                        name: 'Grammar & CoLA',
                        model: 'textattack/roberta-base-CoLA',
                        desc: 'Linguistic acceptability check flagging syntactic and capitalization errors.',
                        badge: 'Syntax',
                        color: '#F97316',
                      },
                      {
                        step: '05',
                        name: 'Logical Coherence (NLI)',
                        model: 'cross-encoder/nli-deberta-v3-base',
                        desc: 'Bidirectional entailment & contradiction check to prevent hallucinated facts.',
                        badge: 'Logic',
                        color: '#A855F7',
                      },
                      {
                        step: '06',
                        name: 'Feedback Synthesis',
                        model: 'Qwen/Qwen2.5-3B-Instruct',
                        desc: 'Synthesizes scores into pedagogical explanations and actionable advice.',
                        badge: 'Reasoning',
                        color: '#F04F9B',
                      },
                    ].map((item) => (
                      <Card
                        key={item.step}
                        className="rounded-2xl border-[#E5E7EB] bg-white p-5 flex flex-col justify-between hover:border-[#18181B] transition-all group"
                      >
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-semibold tabular-nums text-[#71717A]">
                              {item.step}
                            </span>
                            <Badge
                              variant="outline"
                              className="rounded-full text-[10px] font-semibold"
                              style={{ borderColor: item.color, color: item.color }}
                            >
                              {item.badge}
                            </Badge>
                          </div>
                          <h3 className="text-sm font-semibold text-[#18181B] mt-1">
                            {item.name}
                          </h3>
                          <p className="text-xs font-normal text-[#71717A] leading-relaxed">
                            {item.desc}
                          </p>
                        </div>
                        <div className="mt-4 pt-3 border-t border-[#E5E7EB] text-[10px] font-normal text-[#71717A] truncate font-mono">
                          {item.model}
                        </div>
                      </Card>
                    ))}
                  </div>
                </div>

              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB: SINGLE GRADER (Reference C: Workspace & Floating Dock)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'single' && (
              <div className="flex flex-col gap-6 animate-fade-up pb-28">

                {/* Workspace Heading */}
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h2 className="text-xl font-semibold text-[#18181B] tracking-tight">
                      Grader Workspace
                    </h2>
                    <p className="text-xs font-normal text-[#71717A] mt-0.5">
                      Enter the target question and student response to perform comprehensive 5-criterion grading.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-normal text-[#71717A] mr-1 hidden sm:inline">Benchmarks:</span>
                    <Button
                      variant={activeBenchmark === 'photosynthesis' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleLoadSample('photosynthesis')}
                      className={`rounded-full text-xs font-semibold ${
                        activeBenchmark === 'photosynthesis'
                          ? 'bg-[#18181B] text-white shadow-xs'
                          : 'hover:border-[#18181B]'
                      }`}
                    >
                      🌿 Photosynthesis
                    </Button>
                    <Button
                      variant={activeBenchmark === 'watercycle' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleLoadSample('watercycle')}
                      className={`rounded-full text-xs font-semibold ${
                        activeBenchmark === 'watercycle'
                          ? 'bg-[#18181B] text-white shadow-xs'
                          : 'hover:border-[#18181B]'
                      }`}
                    >
                      💧 Water Cycle
                    </Button>
                    <Button
                      variant={activeBenchmark === 'contradiction' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleLoadSample('contradiction')}
                      className={`rounded-full text-xs font-semibold ${
                        activeBenchmark === 'contradiction'
                          ? 'bg-rose-700 hover:bg-rose-800 text-white shadow-xs'
                          : 'text-rose-700 bg-rose-50/50 border-rose-200 hover:bg-rose-50'
                      }`}
                    >
                      ⚠️ Contradiction Test
                    </Button>
                    <Button
                      variant={activeBenchmark === 'informal' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleLoadSample('informal')}
                      className={`rounded-full text-xs font-semibold ${
                        activeBenchmark === 'informal'
                          ? 'bg-amber-600 hover:bg-amber-700 text-white shadow-xs'
                          : 'text-amber-700 bg-amber-50/50 border-amber-200 hover:bg-amber-50'
                      }`}
                    >
                      ✍️ Slang Test
                    </Button>
                  </div>
                </div>

                {/* Two Column Layout: Form & Real-time Canvas */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                  
                  {/* Left: Input Form Card */}
                  <div className="lg:col-span-7 flex flex-col gap-5">
                    <Card className="rounded-2xl border-[#E5E7EB] bg-white shadow-xs">
                      <CardHeader className="pb-3 border-b border-[#E5E7EB]">
                        <CardTitle className="text-sm font-semibold text-[#18181B]">
                          Input Prompt & Student Response
                        </CardTitle>
                        <CardDescription className="text-xs font-normal text-[#71717A]">
                          Provide reference benchmarks to grade student submissions against.
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="p-5 flex flex-col gap-4">
                        
                        {/* Context */}
                        <div className="flex flex-col gap-1.5">
                          <div className="flex items-center justify-between">
                            <Label className="text-xs font-semibold text-[#18181B]">
                              Background Context <span className="text-[#71717A] font-normal">(Optional)</span>
                            </Label>
                            {singleContext && (
                              <button
                                type="button"
                                onClick={() => setSingleContext('')}
                                className="text-[11px] font-normal text-[#71717A] hover:text-rose-600 transition-colors cursor-pointer"
                              >
                                Clear
                              </button>
                            )}
                          </div>
                          <Textarea
                            rows={2}
                            value={singleContext}
                            onChange={(e) => setSingleContext(e.target.value)}
                            placeholder="Optional textbook passage, chapter context, or background..."
                            className="rounded-xl border-[#E5E7EB] text-xs font-normal resize-none"
                          />
                        </div>

                        {/* Question */}
                        <div className="flex flex-col gap-1.5">
                          <Label className="text-xs font-semibold text-[#18181B]">
                            Target Question <span className="text-rose-500 font-semibold">*</span>
                          </Label>
                          <Textarea
                            rows={2}
                            value={singleQuestion}
                            onChange={(e) => setSingleQuestion(e.target.value)}
                            placeholder="e.g. Explain photosynthesis and its primary biological importance..."
                            className="rounded-xl border-[#E5E7EB] text-xs font-normal resize-none"
                            required
                          />
                        </div>

                        {/* Reference Answer */}
                        <div className="flex flex-col gap-1.5">
                          <div className="flex items-center justify-between">
                            <Label className="text-xs font-semibold text-[#18181B]">
                              Golden Reference Answer <span className="text-rose-500 font-semibold">*</span>
                            </Label>
                            <span className="text-[10px] font-normal text-[#71717A]">
                              {singleReference.trim().split(/\s+/).filter(Boolean).length} words
                            </span>
                          </div>
                          <Textarea
                            rows={3}
                            value={singleReference}
                            onChange={(e) => setSingleReference(e.target.value)}
                            placeholder="Ideal instructor model answer..."
                            className="rounded-xl border-[#E5E7EB] text-xs font-normal resize-none"
                            required
                          />
                        </div>

                        {/* Student Answer */}
                        <div className="flex flex-col gap-1.5">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Label className="text-xs font-semibold text-[#18181B]">
                                Student Submission <span className="text-rose-500 font-semibold">*</span>
                              </Label>
                              <kbd className="text-[10px] font-mono font-normal text-[#71717A] bg-[#F7F7F8] px-1.5 py-0.5 rounded border border-[#E5E7EB]">
                                Ctrl ↵
                              </kbd>
                            </div>
                            <span className="text-[10px] font-normal text-[#71717A]">
                              {singleStudent.trim().split(/\s+/).filter(Boolean).length} words · {singleStudent.length} chars
                            </span>
                          </div>
                          <Textarea
                            rows={4}
                            value={singleStudent}
                            onChange={(e) => setSingleStudent(e.target.value)}
                            onKeyDown={(e) => {
                              if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                                e.preventDefault();
                                handleGradeSingle();
                              }
                            }}
                            placeholder="Student answer to grade... (Press Ctrl+Enter to evaluate)"
                            className="rounded-xl border-[#E5E7EB] text-xs font-normal resize-none"
                            required
                          />
                        </div>

                        {singleError && (
                          <Alert variant="destructive" className="rounded-xl">
                            <XCircle className="size-4" strokeWidth={2.5} />
                            <AlertTitle className="text-xs font-semibold">Evaluation Error</AlertTitle>
                            <AlertDescription className="text-xs font-normal">{singleError}</AlertDescription>
                          </Alert>
                        )}
                      </CardContent>
                    </Card>
                  </div>

                  {/* Right: Results Canvas */}
                  <div className="lg:col-span-5 flex flex-col gap-5">
                    {singleResult ? (
                      <div className="flex flex-col gap-4 animate-fade-up">
                        
                        {/* Score Hero Card */}
                        <Card className="rounded-2xl border-[#E5E7EB] bg-white p-6 shadow-xs">
                          <div className="flex items-center gap-6">
                            <ScoreGauge score={singleResult.metrics.final_grade} size={100} />
                            <div className="flex flex-col gap-2 flex-1">
                              <div className="flex items-center gap-2">
                                <Badge
                                  className={`rounded-full text-xs font-semibold ${
                                    getScoreBadgeVariant(singleResult.metrics.final_grade).className
                                  }`}
                                >
                                  {getScoreBadgeVariant(singleResult.metrics.final_grade).label}
                                </Badge>
                              </div>
                              <p className="text-xs font-normal text-[#71717A] leading-relaxed">
                                Weighted composite across 5 NLP criteria.
                              </p>
                              <div className="flex flex-wrap gap-1.5 mt-1">
                                {singleResult.feedback.tags.map((tag, i) => (
                                  <Badge
                                    key={i}
                                    variant="outline"
                                    className={`rounded-full text-[10px] font-semibold flex items-center gap-1 ${getTagBadgeStyle(
                                      tag
                                    )}`}
                                  >
                                    {getTagIcon(tag)}
                                    {tag}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          </div>
                        </Card>

                        {/* Feature Breakdown Progress Bars */}
                        <Card className="rounded-2xl border-[#E5E7EB] bg-white p-5 shadow-xs flex flex-col gap-3.5">
                          <h3 className="text-xs font-semibold text-[#18181B] uppercase tracking-wider">
                            Criterion Breakdown
                          </h3>
                          <FeatureBar
                            label="Semantic Similarity"
                            value={singleResult.metrics.semantic_score}
                            colorHex="#5268F5"
                            weightPct={Math.round(weights.semantic * 100)}
                          />
                          <FeatureBar
                            label="Keyword Coverage"
                            value={singleResult.metrics.coverage_score}
                            colorHex="#0EA5E9"
                            weightPct={Math.round(weights.coverage * 100)}
                          />
                          <FeatureBar
                            label="Academic Formality"
                            value={singleResult.metrics.formality_score}
                            colorHex="#64748B"
                            weightPct={Math.round(weights.formality * 100)}
                          />
                          <FeatureBar
                            label="Grammar & Fluency"
                            value={singleResult.metrics.grammar_score}
                            colorHex="#F97316"
                            weightPct={Math.round(weights.grammar * 100)}
                          />
                          <FeatureBar
                            label="Logical Coherence (NLI)"
                            value={singleResult.metrics.logic_score}
                            colorHex="#A855F7"
                            weightPct={Math.round(weights.logic * 100)}
                            warningBadge={
                              singleResult.metrics.logic_details &&
                              Math.max(
                                singleResult.metrics.logic_details.contradiction || 0,
                                singleResult.metrics.logic_details.backward_contradiction || 0
                              ) > 0.6
                                ? 'Penalty (-40% Cap)'
                                : undefined
                            }
                          />
                        </Card>

                        {/* AI Feedback Card */}
                        <Card className="rounded-2xl border-[#E5E7EB] bg-white p-5 shadow-xs flex flex-col gap-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Sparkles className="size-4 text-[#F04F9B]" strokeWidth={2.5} />
                              <h3 className="text-xs font-semibold text-[#18181B] uppercase tracking-wider">
                                AI Diagnostic Feedback
                              </h3>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={handleCopyFeedback}
                              className="h-7 px-2 text-[11px] font-normal text-[#71717A] hover:text-[#18181B] rounded-lg transition-colors cursor-pointer"
                            >
                              {copiedFeedback ? (
                                <>
                                  <Check className="size-3 text-emerald-600 mr-1" strokeWidth={2.5} />
                                  <span className="text-emerald-600 font-semibold">Copied!</span>
                                </>
                              ) : (
                                <>
                                  <Copy className="size-3 mr-1 text-[#71717A]" strokeWidth={2} />
                                  Copy
                                </>
                              )}
                            </Button>
                          </div>
                          <div className="flex flex-col gap-1.5">
                            <span className="text-[10px] font-semibold uppercase text-[#71717A] tracking-wider">
                              Explanation
                            </span>
                            <p className="text-xs font-normal text-[#18181B] leading-relaxed">
                              {singleResult.feedback.explanation}
                            </p>
                          </div>
                          <Separator className="bg-[#E5E7EB]" />
                          <div className="flex flex-col gap-1.5">
                            <span className="text-[10px] font-semibold uppercase text-[#71717A] tracking-wider">
                              Recommendation
                            </span>
                            <p className="text-xs font-normal text-[#18181B] leading-relaxed">
                              {singleResult.feedback.suggestion}
                            </p>
                          </div>
                        </Card>

                        {/* Missing Keywords (if any) */}
                        {singleResult.metrics.missing_keywords?.length > 0 && (
                          <Card className="rounded-2xl border-rose-200 bg-rose-50/50 p-4">
                            <span className="text-[10px] font-semibold uppercase text-rose-700 tracking-wider">
                              Missing Vocabulary
                            </span>
                            <div className="flex flex-wrap gap-1.5 mt-2">
                              {singleResult.metrics.missing_keywords.map((kw, i) => (
                                <Badge
                                  key={i}
                                  variant="outline"
                                  className="rounded-full text-[10px] font-semibold bg-white text-rose-700 border-rose-200"
                                >
                                  {kw}
                                </Badge>
                              ))}
                            </div>
                          </Card>
                        )}
                      </div>
                    ) : (
                      <Card className="rounded-2xl border-[#E5E7EB] bg-white p-8">
                        <Empty>
                          <EmptyHeader>
                            <EmptyMedia variant="icon">
                              <Wand2 className="size-6 text-[#71717A]" strokeWidth={1.5} />
                            </EmptyMedia>
                            <EmptyTitle className="text-sm font-semibold text-[#18181B]">
                              Ready for Evaluation
                            </EmptyTitle>
                            <EmptyDescription className="text-xs font-normal text-[#71717A]">
                              Review the prompt on the left and click &apos;Run Evaluator&apos; in the floating toolbar below to inspect model scores.
                            </EmptyDescription>
                          </EmptyHeader>
                        </Empty>
                      </Card>
                    )}
                  </div>
                </div>

                {/* ── Reference C: Floating Capsule Control Dock ────────────── */}
                <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 w-fit max-w-[95vw]">
                  <div className="bg-white/95 backdrop-blur-xl border border-[#E5E7EB] rounded-full px-4 py-2 shadow-2xl shadow-zinc-900/10 flex items-center gap-3">
                    
                    {/* Preset Segment Pills */}
                    <div className="flex items-center gap-1 bg-[#F7F7F8] p-1 rounded-full border border-[#E5E7EB]">
                      {['balanced', 'content_focused', 'logic_heavy', 'quick_check'].map((pKey) => {
                        const active = activePreset === pKey;
                        const label =
                          pKey === 'balanced'
                            ? 'Balanced'
                            : pKey === 'content_focused'
                            ? 'Content'
                            : pKey === 'logic_heavy'
                            ? 'Logic'
                            : 'Quick';
                        return (
                          <button
                            key={pKey}
                            onClick={() => handlePresetChange(pKey)}
                            className={`px-3 py-1 rounded-full text-xs font-semibold transition-all cursor-pointer ${
                              active
                                ? 'bg-white text-[#18181B] shadow-xs'
                                : 'text-[#71717A] hover:text-[#18181B]'
                            }`}
                          >
                            {label}
                          </button>
                        );
                      })}
                    </div>

                    <Separator orientation="vertical" className="h-6 bg-[#E5E7EB]" />

                    {/* Reset Button */}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          type="button"
                          onClick={handleClearWorkspace}
                          className="size-8 rounded-full flex items-center justify-center text-[#71717A] hover:text-[#18181B] hover:bg-[#F7F7F8] transition-colors cursor-pointer"
                          aria-label="Reset workspace inputs"
                        >
                          <RotateCcw className="size-3.5" strokeWidth={2} />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent side="top">
                        <span className="text-xs font-normal">Clear workspace inputs</span>
                      </TooltipContent>
                    </Tooltip>

                    {/* Primary Run Action Pill */}
                    <Button
                      onClick={() => handleGradeSingle()}
                      disabled={isGradingSingle}
                      className="rounded-full bg-[#18181B] hover:bg-[#181113] text-white font-semibold text-xs px-5 py-2 h-auto cursor-pointer shadow-sm active:scale-95 transition-all"
                    >
                      {isGradingSingle ? (
                        <>
                          <Spinner className="size-3.5 mr-1.5" />
                          Evaluating...
                        </>
                      ) : (
                        <>
                          <Play className="size-3.5 mr-1.5 fill-current" strokeWidth={2.5} />
                          <span>Run Evaluator</span>
                          <kbd className="ml-2 px-1.5 py-0.5 rounded bg-zinc-800 text-[10px] text-zinc-300 font-mono font-normal">
                            Ctrl ↵
                          </kbd>
                        </>
                      )}
                    </Button>
                  </div>
                </div>

              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB: BATCH GRADER (Reference B: Application Data Dashboard)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'batch' && (
              <div className="flex flex-col gap-6 animate-fade-up">

                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h2 className="text-xl font-semibold text-[#18181B] tracking-tight">
                      Batch Student Assessment
                    </h2>
                    <p className="text-xs font-normal text-[#71717A] mt-0.5">
                      Process multiple student responses simultaneously with uniform rubrics.
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleLoadSample('photosynthesis')}
                      className="rounded-full text-xs font-semibold"
                    >
                      Load Sample Data
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleExportCSV}
                      disabled={!batchItems.length}
                      className="rounded-full text-xs font-semibold"
                    >
                      <Download className="size-3.5 mr-1.5" strokeWidth={2} />
                      Export CSV
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                  
                  {/* Left: Batch Config */}
                  <div className="lg:col-span-5 flex flex-col gap-4">
                    <Card className="rounded-2xl border-[#E5E7EB] bg-white shadow-xs p-5 flex flex-col gap-4">
                      <div className="flex flex-col gap-1.5">
                        <Label className="text-xs font-semibold text-[#18181B]">Question Title</Label>
                        <Input
                          value={batchQuestion}
                          onChange={(e) => setBatchQuestion(e.target.value)}
                          className="rounded-xl border-[#E5E7EB] text-xs font-normal"
                        />
                      </div>

                      <div className="flex flex-col gap-1.5">
                        <Label className="text-xs font-semibold text-[#18181B]">Reference Answer</Label>
                        <Textarea
                          rows={2}
                          value={batchReference}
                          onChange={(e) => setBatchReference(e.target.value)}
                          className="rounded-xl border-[#E5E7EB] text-xs font-normal resize-none"
                        />
                      </div>

                      <div className="flex flex-col gap-1.5">
                        <div className="flex justify-between items-center">
                          <Label className="text-xs font-semibold text-[#18181B]">
                            Student Responses
                          </Label>
                          <span className="text-[10px] font-normal text-[#71717A]">
                            One per line (`Name: answer`)
                          </span>
                        </div>
                        <Textarea
                          rows={8}
                          value={batchRawInput}
                          onChange={(e) => setBatchRawInput(e.target.value)}
                          className="rounded-xl border-[#E5E7EB] text-xs font-mono font-normal resize-none"
                        />
                      </div>

                      <Button
                        onClick={handleGradeBatch}
                        disabled={isGradingBatch || !batchRawInput.trim()}
                        className="rounded-full bg-[#18181B] hover:bg-[#181113] text-white font-semibold text-xs py-2.5 h-auto cursor-pointer"
                      >
                        {isGradingBatch ? (
                          <>
                            <Spinner className="size-3.5 mr-2" />
                            Evaluating Batch...
                          </>
                        ) : (
                          <>
                            <Play className="size-3.5 mr-2 fill-current" strokeWidth={2.5} />
                            Evaluate All Responses
                          </>
                        )}
                      </Button>
                    </Card>
                  </div>

                  {/* Right: Data Table & Inspection */}
                  <div className="lg:col-span-7 flex flex-col gap-5">
                    {batchItems.length > 0 ? (() => {
                      const countAll = batchItems.length;
                      const countCorrect = batchItems.filter((i) => i.result?.feedback.tags.includes('Correct')).length;
                      const countPartial = batchItems.filter((i) => i.result?.feedback.tags.includes('Partially Correct')).length;
                      const countIncorrect = batchItems.filter(
                        (i) => i.result && !i.result.feedback.tags.includes('Correct') && !i.result.feedback.tags.includes('Partially Correct')
                      ).length;

                      const filteredBatchItems = batchItems.filter((item) => {
                        if (batchSearch.trim()) {
                          const q = batchSearch.toLowerCase();
                          const matchId = item.id.toLowerCase().includes(q);
                          const matchText = item.student.toLowerCase().includes(q);
                          if (!matchId && !matchText) return false;
                        }
                        if (batchFilter === 'all') return true;
                        if (!item.result) return true;
                        const tags = item.result.feedback.tags;
                        if (batchFilter === 'correct') return tags.includes('Correct');
                        if (batchFilter === 'partial') return tags.includes('Partially Correct');
                        if (batchFilter === 'incorrect') return !tags.includes('Correct') && !tags.includes('Partially Correct');
                        return true;
                      });

                      return (
                        <div className="flex flex-col gap-4">
                          {/* Search & Filter Toolbar */}
                          <div className="flex flex-wrap items-center justify-between gap-3">
                            <div className="flex items-center gap-1 bg-[#F7F7F8] p-1 rounded-full border border-[#E5E7EB]">
                              {[
                                { key: 'all' as const, label: 'All', count: countAll },
                                { key: 'correct' as const, label: 'Correct', count: countCorrect },
                                { key: 'partial' as const, label: 'Partial', count: countPartial },
                                { key: 'incorrect' as const, label: 'Capped', count: countIncorrect },
                              ].map((f) => {
                                const active = batchFilter === f.key;
                                return (
                                  <button
                                    key={f.key}
                                    type="button"
                                    onClick={() => setBatchFilter(f.key)}
                                    className={`px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                                      active
                                        ? 'bg-white text-[#18181B] shadow-xs'
                                        : 'text-[#71717A] hover:text-[#18181B]'
                                    }`}
                                  >
                                    <span>{f.label}</span>
                                    <span className="text-[10px] font-normal px-1.5 py-0.2 rounded-full bg-[#E5E7EB]/60 text-[#18181B]">
                                      {f.count}
                                    </span>
                                  </button>
                                );
                              })}
                            </div>

                            <div className="relative w-full sm:w-52">
                              <Search className="size-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[#71717A]" />
                              <Input
                                placeholder="Search submissions..."
                                value={batchSearch}
                                onChange={(e) => setBatchSearch(e.target.value)}
                                className="h-8 pl-8 pr-3 text-xs rounded-full border-[#E5E7EB] bg-white font-normal"
                              />
                            </div>
                          </div>

                          <Card className="rounded-2xl border-[#E5E7EB] bg-white overflow-hidden shadow-xs">
                            <Table>
                              <TableHeader className="bg-[#F7F7F8]">
                                <TableRow className="border-[#E5E7EB]">
                                  <TableHead className="text-xs font-semibold text-[#71717A]">Student</TableHead>
                                  <TableHead className="text-xs font-semibold text-[#71717A]">Response Preview</TableHead>
                                  <TableHead className="text-xs font-semibold text-[#71717A] text-right">Grade</TableHead>
                                  <TableHead className="text-xs font-semibold text-[#71717A]">Status Tag</TableHead>
                                  <TableHead className="w-10" />
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {filteredBatchItems.map((item) => {
                                  const grade = item.result?.metrics.final_grade;
                                  const tag = item.result?.feedback.tags[0] || '';
                                  const isSelected = selectedBatchItem?.id === item.id;
                                  return (
                                    <TableRow
                                      key={item.id}
                                      onClick={() => item.result && setSelectedBatchItem(item)}
                                      className={`border-[#E5E7EB] cursor-pointer transition-colors ${
                                        isSelected ? 'bg-[#F7F7F8]' : 'hover:bg-[#F7F7F8]/50'
                                      }`}
                                    >
                                      <TableCell className="text-xs font-semibold text-[#18181B]">
                                        {item.id}
                                      </TableCell>
                                      <TableCell className="text-xs font-normal text-[#71717A] max-w-[200px] truncate">
                                        {item.student}
                                      </TableCell>
                                      <TableCell className="text-xs font-semibold tabular-nums text-right text-[#18181B]">
                                        {item.status === 'grading' ? (
                                          <Spinner className="size-3 inline" />
                                        ) : item.status === 'success' && grade !== undefined ? (
                                          `${Math.round(grade)}%`
                                        ) : (
                                          <span className="text-rose-500 font-semibold">Err</span>
                                        )}
                                      </TableCell>
                                      <TableCell>
                                        {item.status === 'success' && tag && (
                                          <Badge
                                            variant="outline"
                                            className={`rounded-full text-[10px] font-semibold ${getTagBadgeStyle(
                                              tag
                                            )}`}
                                          >
                                            {tag}
                                          </Badge>
                                        )}
                                      </TableCell>
                                      <TableCell>
                                        <ChevronRight
                                          className={`size-4 text-[#71717A] ${
                                            isSelected ? 'text-[#18181B]' : ''
                                          }`}
                                          strokeWidth={2}
                                        />
                                      </TableCell>
                                    </TableRow>
                                  );
                                })}
                              </TableBody>
                            </Table>
                          </Card>

                          {/* Selected Student Detail Card */}
                          {selectedBatchItem?.result && (
                            <Card className="rounded-2xl border-[#E5E7EB] bg-white p-5 shadow-xs flex flex-col gap-4 animate-fade-up">
                              <div className="flex flex-wrap items-center justify-between border-b border-[#E5E7EB] pb-3 gap-2">
                                <div className="flex items-center gap-2">
                                  <span className="text-sm font-semibold text-[#18181B]">
                                    Inspection: {selectedBatchItem.id}
                                  </span>
                                  <Badge
                                    className={`rounded-full text-[10px] font-semibold ${
                                      getScoreBadgeVariant(selectedBatchItem.result.metrics.final_grade).className
                                    }`}
                                  >
                                    {Math.round(selectedBatchItem.result.metrics.final_grade)}%
                                  </Badge>
                                </div>
                                <div className="flex items-center gap-2">
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => handleInspectBatchInWorkspace(selectedBatchItem)}
                                    className="rounded-full text-xs font-semibold h-7 px-3"
                                  >
                                    Open in Workspace
                                    <ArrowRight className="size-3 ml-1" />
                                  </Button>
                                </div>
                              </div>

                              {/* 5-Criteria Mini Breakdown */}
                              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                                {[
                                  { name: 'Semantic', val: selectedBatchItem.result.metrics.semantic_score, color: '#5268F5' },
                                  { name: 'Coverage', val: selectedBatchItem.result.metrics.coverage_score, color: '#0EA5E9' },
                                  { name: 'Formality', val: selectedBatchItem.result.metrics.formality_score, color: '#64748B' },
                                  { name: 'Grammar', val: selectedBatchItem.result.metrics.grammar_score, color: '#F97316' },
                                  { name: 'Logic', val: selectedBatchItem.result.metrics.logic_score, color: '#A855F7' },
                                ].map((c) => (
                                  <div key={c.name} className="p-2.5 rounded-xl bg-[#F7F7F8] border border-[#E5E7EB] flex flex-col gap-1">
                                    <span className="text-[10px] font-semibold text-[#71717A] uppercase">{c.name}</span>
                                    <span className="text-xs font-semibold tabular-nums text-[#18181B]">
                                      {Math.round(c.val * 100)}%
                                    </span>
                                    <div className="h-1 w-full rounded-full bg-zinc-200 overflow-hidden">
                                      <div className="h-full rounded-full" style={{ width: `${Math.round(c.val * 100)}%`, backgroundColor: c.color }} />
                                    </div>
                                  </div>
                                ))}
                              </div>

                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-normal">
                                <div className="flex flex-col gap-1">
                                  <span className="font-semibold text-[#18181B]">Explanation</span>
                                  <p className="text-[#71717A] leading-relaxed">
                                    {selectedBatchItem.result.feedback.explanation}
                                  </p>
                                </div>
                                <div className="flex flex-col gap-1">
                                  <span className="font-semibold text-[#18181B]">Recommendation</span>
                                  <p className="text-[#71717A] leading-relaxed">
                                    {selectedBatchItem.result.feedback.suggestion}
                                  </p>
                                </div>
                              </div>
                            </Card>
                          )}
                        </div>
                      );
                    })() : (
                      <Card className="rounded-2xl border-[#E5E7EB] bg-white p-8">
                        <Empty>
                          <EmptyHeader>
                            <EmptyMedia variant="icon">
                              <FileSpreadsheet className="size-6 text-[#71717A]" strokeWidth={1.5} />
                            </EmptyMedia>
                            <EmptyTitle className="text-sm font-semibold text-[#18181B]">
                              No Batch Data
                            </EmptyTitle>
                            <EmptyDescription className="text-xs font-normal text-[#71717A]">
                              Click &apos;Load Sample Data&apos; or type student answers on the left to execute multi-response evaluations.
                            </EmptyDescription>
                          </EmptyHeader>
                        </Empty>
                      </Card>
                    )}
                  </div>
                </div>

              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB: RESEARCH LAB (Reference B & C: Ablation & Models)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'research' && (
              <div className="flex flex-col gap-6 animate-fade-up">

                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-semibold text-[#18181B] tracking-tight flex items-center gap-2">
                      <FlaskConical className="size-5 text-[#5268F5]" strokeWidth={2.5} />
                      Research Lab & Ablation Benchmark
                    </h2>
                    <p className="text-xs font-normal text-[#71717A] mt-0.5">
                      Verify individual component contributions via Wilcoxon signed-rank tests and export publication-ready LaTeX tables.
                    </p>
                  </div>
                  <Badge
                    variant="outline"
                    className="rounded-full text-xs px-3 py-1 font-semibold"
                  >
                    {isLive ? 'Live Research API' : 'Simulated Research Mode'}
                  </Badge>
                </div>

                {/* Active Architecture Summary */}
                <Card className="rounded-2xl border-[#E5E7EB] bg-white p-5 shadow-xs">
                  <h3 className="text-xs font-semibold text-[#18181B] uppercase tracking-wider mb-3">
                    Active Grader Backbones
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
                    {[
                      {
                        name: 'Semantic',
                        val: researchConfig?.semantic_model?.split('/').pop() || 'sup-simcse-roberta-large',
                      },
                      {
                        name: 'Keyword',
                        val: researchConfig?.keybert_model || 'all-MiniLM-L6-v2',
                      },
                      {
                        name: 'Formality',
                        val: researchConfig?.formality_model?.split('/').pop() || 'roberta-base-formality',
                      },
                      {
                        name: 'Grammar',
                        val: researchConfig?.grammar_model?.split('/').pop() || 'roberta-base-CoLA',
                      },
                      {
                        name: 'Logic NLI',
                        val: researchConfig?.logic_model?.split('/').pop() || 'nli-deberta-v3-base',
                      },
                      {
                        name: 'LLM Synth',
                        val: researchConfig?.reasoning_model?.split('/').pop() || 'Qwen2.5-3B-Instruct',
                      },
                    ].map((m) => (
                      <div
                        key={m.name}
                        className="p-3 rounded-xl bg-[#F7F7F8] border border-[#E5E7EB] flex flex-col gap-1"
                      >
                        <span className="text-[10px] font-semibold text-[#71717A] uppercase">
                          {m.name}
                        </span>
                        <span className="text-xs font-semibold text-[#18181B] truncate font-mono">
                          {m.val}
                        </span>
                      </div>
                    ))}
                  </div>
                </Card>

                {/* Ablation Runner Controls */}
                <Card className="rounded-2xl border-[#E5E7EB] bg-white p-6 shadow-xs flex flex-col gap-5">
                  <div className="flex flex-wrap items-end gap-4">
                    <div className="flex-1 min-w-[200px] flex flex-col gap-1.5">
                      <Label className="text-xs font-semibold text-[#18181B]">Feature Cache Path</Label>
                      <Input
                        value={ablationCachePath}
                        onChange={(e) => setAblationCachePath(e.target.value)}
                        placeholder="./cache/features.pkl"
                        className="rounded-xl border-[#E5E7EB] text-xs font-mono font-normal"
                      />
                    </div>
                    <div className="w-28 flex flex-col gap-1.5">
                      <Label className="text-xs font-semibold text-[#18181B]">Seed</Label>
                      <Input
                        type="number"
                        value={ablationSeed}
                        onChange={(e) => setAblationSeed(parseInt(e.target.value) || 42)}
                        className="rounded-xl border-[#E5E7EB] text-xs font-mono font-normal"
                      />
                    </div>
                    <Button
                      onClick={handleRunAblation}
                      disabled={isRunningAblation}
                      className="rounded-full bg-[#18181B] hover:bg-[#181113] text-white font-semibold text-xs px-6 py-2.5 h-auto cursor-pointer"
                    >
                      {isRunningAblation ? (
                        <>
                          <Spinner className="size-3.5 mr-2" />
                          Running Ablation...
                        </>
                      ) : (
                        <>
                          <FlaskConical className="size-3.5 mr-2" strokeWidth={2.5} />
                          Run Ablation Study
                        </>
                      )}
                    </Button>
                  </div>

                  {ablationError && (
                    <Alert variant="destructive" className="rounded-xl">
                      <XCircle className="size-4" strokeWidth={2.5} />
                      <AlertTitle className="text-xs font-semibold">Error</AlertTitle>
                      <AlertDescription className="text-xs font-normal">{ablationError}</AlertDescription>
                    </Alert>
                  )}

                  {/* Ablation Results Table */}
                  {ablationResults.length > 0 && (
                    <div className="flex flex-col gap-4 animate-fade-up">
                      <div className="rounded-2xl border border-[#E5E7EB] overflow-hidden">
                        <Table>
                          <TableHeader className="bg-[#F7F7F8]">
                            <TableRow className="border-[#E5E7EB]">
                              <TableHead className="text-xs font-semibold text-[#71717A]">Variant</TableHead>
                              <TableHead className="text-xs font-semibold text-[#71717A]">Accuracy</TableHead>
                              <TableHead className="text-xs font-semibold text-[#71717A]">Macro F1</TableHead>
                              <TableHead className="text-xs font-semibold text-[#71717A]">QWK</TableHead>
                              <TableHead className="text-xs font-semibold text-[#71717A]">Δ F1</TableHead>
                              <TableHead className="text-xs font-semibold text-[#71717A]">Δ QWK</TableHead>
                              <TableHead className="text-xs font-semibold text-[#71717A]">Sig.</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {ablationResults.map((row, i) => {
                              const isFull = row.Variant.includes('Full Model');
                              return (
                                <TableRow
                                  key={i}
                                  className={`border-[#E5E7EB] ${
                                    isFull ? 'bg-[#F7F7F8] font-semibold' : 'hover:bg-[#F7F7F8]/50'
                                  }`}
                                >
                                  <TableCell className="text-xs font-semibold text-[#18181B]">
                                    {row.Variant}
                                  </TableCell>
                                  <TableCell className="text-xs font-mono font-normal text-[#18181B] tabular-nums">
                                    {row.Accuracy}
                                  </TableCell>
                                  <TableCell className="text-xs font-mono font-normal text-[#18181B] tabular-nums">
                                    {row['Macro F1']}
                                  </TableCell>
                                  <TableCell className="text-xs font-mono font-normal text-[#18181B] tabular-nums">
                                    {row.QWK}
                                  </TableCell>
                                  <TableCell className="text-xs font-mono font-semibold text-rose-600 tabular-nums">
                                    {row['Δ F1']}
                                  </TableCell>
                                  <TableCell className="text-xs font-mono font-semibold text-rose-600 tabular-nums">
                                    {row['Δ QWK']}
                                  </TableCell>
                                  <TableCell>
                                    {row['Sig.'] && (
                                      <Badge
                                        variant="outline"
                                        className="rounded-full text-[10px] font-semibold bg-rose-50 text-rose-700 border-rose-200"
                                      >
                                        {row['Sig.']}
                                      </Badge>
                                    )}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>

                      {/* LaTeX Copy Panel */}
                      {ablationLatex && (
                        <Card className="rounded-2xl border-[#E5E7EB] bg-[#181113] p-5 text-white flex flex-col gap-3">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <span className="text-xs font-semibold text-white/80 flex items-center gap-1.5">
                              <Table2 className="size-3.5" strokeWidth={2} />
                              LaTeX Table (Academic Publication Snippet)
                            </span>
                            <div className="flex items-center gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={downloadAblationJson}
                                className="rounded-full bg-white/10 hover:bg-white/20 text-white border-white/20 text-xs font-semibold h-7"
                              >
                                <Download className="size-3 mr-1" strokeWidth={2} />
                                Export JSON
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={downloadLatex}
                                className="rounded-full bg-white/10 hover:bg-white/20 text-white border-white/20 text-xs font-semibold h-7"
                              >
                                <FileDown className="size-3 mr-1" strokeWidth={2} />
                                Download .tex
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={copyLatex}
                                className="rounded-full bg-white/10 hover:bg-white/20 text-white border-white/20 text-xs font-semibold h-7"
                              >
                                {copiedLatex ? (
                                  <>
                                    <Check className="size-3 mr-1" strokeWidth={2.5} /> Copied!
                                  </>
                                ) : (
                                  <>
                                    <Copy className="size-3 mr-1" strokeWidth={2} /> Copy LaTeX
                                  </>
                                )}
                              </Button>
                            </div>
                          </div>
                          <Textarea
                            readOnly
                            rows={5}
                            value={ablationLatex}
                            className="rounded-xl bg-black/40 border-white/10 text-[11px] font-mono text-emerald-300 resize-none"
                          />
                          <div className="text-[10px] font-normal text-white/50 flex flex-wrap items-center gap-3 pt-1 border-t border-white/10">
                            <span>Significance:</span>
                            <span>*** p &lt; 0.001</span>
                            <span>** p &lt; 0.01</span>
                            <span>* p &lt; 0.05</span>
                            <span>ns: not significant</span>
                          </div>
                        </Card>
                      )}
                    </div>
                  )}
                </Card>

              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB: SYSTEM SETTINGS (Reference B)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'settings' && (
              <div className="flex flex-col gap-6 max-w-4xl mx-auto animate-fade-up">

                <div>
                  <h2 className="text-xl font-semibold text-[#18181B] tracking-tight">
                    System Configuration
                  </h2>
                  <p className="text-xs font-normal text-[#71717A] mt-0.5">
                    Adjust endpoint bindings, grading criterion weights, and threshold bounds.
                  </p>
                </div>

                {/* API Binding */}
                <Card className="rounded-2xl border-[#E5E7EB] bg-white p-5 shadow-xs flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <Database className="size-4 text-[#5268F5]" strokeWidth={2} />
                    <h3 className="text-sm font-semibold text-[#18181B]">FastAPI Endpoint Connection</h3>
                  </div>
                  <div className="flex gap-3 items-end">
                    <div className="flex-1 flex flex-col gap-1.5">
                      <Label className="text-xs font-semibold text-[#18181B]">Server URL</Label>
                      <Input
                        value={apiUrl}
                        onChange={(e) => setApiUrl(e.target.value)}
                        className="rounded-xl border-[#E5E7EB] text-xs font-mono font-normal"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      {pingLatency !== null && (
                        <span className="text-xs font-semibold text-emerald-600 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200">
                          ⚡ {pingLatency}ms
                        </span>
                      )}
                      <Button
                        variant="outline"
                        onClick={testPing}
                        className="rounded-full text-xs font-semibold cursor-pointer"
                      >
                        Test Connection
                      </Button>
                    </div>
                  </div>
                </Card>

                {/* Weights Sliders */}
                <Card className="rounded-2xl border-[#E5E7EB] bg-white p-6 shadow-xs flex flex-col gap-5">
                  <div className="flex flex-wrap items-center justify-between border-b border-[#E5E7EB] pb-3 gap-2">
                    <div className="flex items-center gap-2">
                      <Sliders className="size-4 text-[#18181B]" strokeWidth={2} />
                      <h3 className="text-sm font-semibold text-[#18181B]">Criterion Scoring Weights</h3>
                      {(() => {
                        const sum = Math.round(
                          (weights.semantic + weights.coverage + weights.formality + weights.grammar + weights.logic) * 100
                        );
                        return (
                          <Badge
                            variant="outline"
                            className={`rounded-full text-[10px] font-semibold ${
                              sum === 100
                                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                : 'bg-amber-50 text-amber-700 border-amber-200'
                            }`}
                          >
                            Sum: {sum}%
                          </Badge>
                        );
                      })()}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-[#71717A]">
                        Preset: <span className="text-[#18181B] capitalize">{activePreset}</span>
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleNormalizeWeights}
                        className="rounded-full text-xs font-semibold h-7 px-3"
                      >
                        Normalize (20% each)
                      </Button>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {[
                      { key: 'semantic', label: 'Semantic Similarity', color: '#5268F5' },
                      { key: 'coverage', label: 'Keyword Coverage', color: '#0EA5E9' },
                      { key: 'formality', label: 'Academic Formality', color: '#64748B' },
                      { key: 'grammar', label: 'Grammar & Fluency', color: '#F97316' },
                      { key: 'logic', label: 'Logical Coherence (NLI)', color: '#A855F7' },
                    ].map(({ key, label, color }) => {
                      const k = key as keyof Weights;
                      const val = Math.round(weights[k] * 100);
                      return (
                        <div key={key} className="flex flex-col gap-2">
                          <div className="flex justify-between items-center text-xs">
                            <span className="font-semibold text-[#18181B]">{label}</span>
                            <span className="font-semibold tabular-nums text-[#18181B]">{val}%</span>
                          </div>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={weights[k]}
                            onChange={(e) => handleWeightChange(k, parseFloat(e.target.value))}
                            className="w-full cursor-pointer"
                            style={{
                              background: `linear-gradient(to right, ${color} ${val}%, #E5E7EB ${val}%)`,
                            }}
                          />
                        </div>
                      );
                    })}
                  </div>
                </Card>

                {/* Thresholds Reference */}
                <Card className="rounded-2xl border-[#E5E7EB] bg-white p-5 shadow-xs flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <Shield className="size-4 text-[#18181B]" strokeWidth={2} />
                    <h3 className="text-sm font-semibold text-[#18181B]">Scoring Thresholds Reference</h3>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-normal text-[#71717A]">
                    <div className="p-3 bg-[#F7F7F8] rounded-xl border border-[#E5E7EB] flex flex-col gap-1">
                      <span className="font-semibold text-[#18181B]">Semantic Pass</span>
                      <span>Correct: ≥ {thresholds.semantic_correct}</span>
                      <span>Partial: ≥ {thresholds.semantic_partial}</span>
                    </div>
                    <div className="p-3 bg-[#F7F7F8] rounded-xl border border-[#E5E7EB] flex flex-col gap-1">
                      <span className="font-semibold text-[#18181B]">Keyword Coverage</span>
                      <span>Target: ≥ {thresholds.coverage_correct}</span>
                      <span>Missing: &lt; {thresholds.coverage_missing}</span>
                    </div>
                    <div className="p-3 bg-[#F7F7F8] rounded-xl border border-[#E5E7EB] flex flex-col gap-1">
                      <span className="font-semibold text-[#18181B]">NLI Contradiction</span>
                      <span>High Cap (40%): ≥ {thresholds.contradiction_high}</span>
                      <span>Moderate: ≥ {thresholds.contradiction_moderate}</span>
                    </div>
                  </div>
                </Card>

              </div>
            )}

          </main>
        </div>
      </div>
    </TooltipProvider>
  );
}
