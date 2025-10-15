//! # Butterfly Schedule
//!
//! Task scheduling and execution coordination for distributed inference.
//! Manages the pipeline of inference requests across partitioned model nodes.

use butterfly_core::{ModelPartition, NodeId};
use std::collections::VecDeque;

/// Represents a single inference request in the system
#[derive(Debug, Clone)]
pub struct InferenceTask {
    pub task_id: u64,
    pub input: Vec<f32>,
    pub priority: u32,
}

/// Status of a task in the execution pipeline
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Trait for scheduling strategies that determine task execution order
pub trait Scheduler: Send + Sync {
    /// Add a new task to the scheduler
    fn enqueue(&mut self, task: InferenceTask);

    /// Get the next task to execute on the given node
    fn next_task(&mut self, node_id: NodeId) -> Option<InferenceTask>;

    /// Mark a task as completed
    fn complete_task(&mut self, task_id: u64);

    /// Get the current status of a task
    fn task_status(&self, task_id: u64) -> Option<TaskStatus>;
}

/// Simple FIFO scheduler implementation
pub struct FifoScheduler {
    queue: VecDeque<InferenceTask>,
    running: Vec<u64>,
    completed: Vec<u64>,
}

impl FifoScheduler {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            running: Vec::new(),
            completed: Vec::new(),
        }
    }
}

impl Default for FifoScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler for FifoScheduler {
    fn enqueue(&mut self, task: InferenceTask) {
        self.queue.push_back(task);
    }

    fn next_task(&mut self, _node_id: NodeId) -> Option<InferenceTask> {
        if let Some(task) = self.queue.pop_front() {
            self.running.push(task.task_id);
            Some(task)
        } else {
            None
        }
    }

    fn complete_task(&mut self, task_id: u64) {
        self.running.retain(|&id| id != task_id);
        self.completed.push(task_id);
    }

    fn task_status(&self, task_id: u64) -> Option<TaskStatus> {
        if self.completed.contains(&task_id) {
            Some(TaskStatus::Completed)
        } else if self.running.contains(&task_id) {
            Some(TaskStatus::Running)
        } else if self.queue.iter().any(|t| t.task_id == task_id) {
            Some(TaskStatus::Pending)
        } else {
            None
        }
    }
}

/// Coordinates execution across multiple partitions
pub struct ExecutionCoordinator {
    partitions: Vec<ModelPartition>,
    scheduler: Box<dyn Scheduler>,
}

impl ExecutionCoordinator {
    pub fn new(partitions: Vec<ModelPartition>, scheduler: Box<dyn Scheduler>) -> Self {
        Self {
            partitions,
            scheduler,
        }
    }

    pub fn submit_task(&mut self, task: InferenceTask) {
        self.scheduler.enqueue(task);
    }

    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fifo_scheduler() {
        let mut scheduler = FifoScheduler::new();

        let task = InferenceTask {
            task_id: 1,
            input: vec![1.0, 2.0, 3.0],
            priority: 0,
        };

        scheduler.enqueue(task.clone());
        assert_eq!(scheduler.task_status(1), Some(TaskStatus::Pending));

        let next = scheduler.next_task(NodeId(0));
        assert!(next.is_some());
        assert_eq!(scheduler.task_status(1), Some(TaskStatus::Running));

        scheduler.complete_task(1);
        assert_eq!(scheduler.task_status(1), Some(TaskStatus::Completed));
    }
}
