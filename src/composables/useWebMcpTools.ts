import { onBeforeUnmount, onMounted } from 'vue'
import type { CharacterStats, StoryOutput } from '../types'

// WebMCP (https://github.com/webmachinelearning/webmcp) lets a page hand tools to an AI agent.
// Only Chrome 149+ and Edge 150+ implement it so far, so without `document.modelContext`
// nothing here runs and the page works as before.

/** The part of the WebMCP API this page uses. */
interface WebMcpTool {
  name: string
  description: string
  inputSchema: Record<string, unknown>
  annotations?: { readOnlyHint?: boolean; untrustedContentHint?: boolean }
  execute: (input: unknown) => Promise<string>
}

interface ModelContext {
  registerTool(tool: WebMcpTool, options: { signal: AbortSignal }): Promise<void>
}

/** What the game screen lets an agent do. */
export interface GameActions {
  /** Plays one turn, exactly like the input box. Resolves to null when the turn failed. */
  takeAction(action: string): Promise<StoryOutput | null>
  readCharacterSheet(): Promise<CharacterStats>
  /** True while a turn is in flight, whoever started it. */
  isBusy(): boolean
  /** The error the game screen shows, if any. */
  lastError(): string | null
}

/** Chrome's tool guidance caps one result at 1.5K characters. */
export const MAX_RESULT_CHARS = 1500

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Chrome before 155 may hand the arguments over as a JSON string, later versions as an object. */
function toArgs(input: unknown): Record<string, unknown> {
  if (typeof input === 'string') {
    try {
      const parsed: unknown = JSON.parse(input)
      return isRecord(parsed) ? parsed : {}
    } catch {
      return {}
    }
  }
  return isRecord(input) ? input : {}
}

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, Math.max(0, max - 3))}...`
}

/** The narration first, then the dice and the suggested moves, which always fit. */
export function formatStory(story: StoryOutput): string {
  const tail: string[] = []
  const rolls = story.dice_rolls ?? []
  if (rolls.length > 0) {
    tail.push(`Dice: ${rolls.map((r) => `${r.dice_type}=${r.result} (${r.reason})`).join(', ')}`)
  }
  const moves = story.actions_suggestions ?? []
  if (moves.length > 0) {
    tail.push(`Suggested next moves: ${moves.join(' | ')}`)
  }
  const tailText = clip(tail.join('\n\n'), MAX_RESULT_CHARS / 2)
  const room = MAX_RESULT_CHARS - (tailText ? tailText.length + 2 : 0)
  const narration = clip((story.response ?? '').trim(), room)
  return tailText ? `${narration}\n\n${tailText}` : narration
}

function gameTools(game: GameActions): WebMcpTool[] {
  return [
    {
      name: 'take_action',
      description:
        "Play the player's next turn in this adventure: one thing they say or do, in their own words, " +
        "for example 'I search the tavern for a quest board'. Returns the Game Master's narration, " +
        'any dice rolls and suggested next moves. The turn also appears in the game chat.',
      inputSchema: {
        type: 'object',
        properties: {
          action: { type: 'string', description: 'What the player says or does this turn, in plain words.' },
        },
        required: ['action'],
      },
      annotations: { untrustedContentHint: true },
      async execute(input) {
        const action = toArgs(input).action
        if (typeof action !== 'string' || action.trim() === '') {
          return "Pass the player's move as a non-empty string in \"action\", for example \"I search the tavern for a quest board\"."
        }
        if (game.isBusy()) {
          return 'The Game Master is still answering the last turn. Wait a moment, then try again.'
        }
        const story = await game.takeAction(action.trim())
        if (story === null) {
          return `The Game Master could not answer (${game.lastError() ?? 'unknown error'}). Check that the Chapter 5 servers are running, then try again.`
        }
        return formatStory(story)
      },
    },
    {
      name: 'read_character_sheet',
      description:
        "Read the player's character sheet: name, race, class, level, experience, ability scores and inventory.",
      inputSchema: { type: 'object', properties: {} },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      async execute() {
        try {
          return clip(JSON.stringify(await game.readCharacterSheet()), MAX_RESULT_CHARS)
        } catch (e) {
          const reason = e instanceof Error ? e.message : 'unknown error'
          return `Could not read the character sheet (${reason}). Check that the Game Master is running.`
        }
      },
    },
  ]
}

/** Registers the game tools until `signal` aborts. Returns false when the browser has no WebMCP. */
export function registerGameTools(game: GameActions, signal: AbortSignal): boolean {
  const modelContext = (document as Document & { modelContext?: ModelContext }).modelContext
  if (!modelContext) {
    return false
  }
  for (const tool of gameTools(game)) {
    modelContext.registerTool(tool, { signal }).catch((e: unknown) => {
      console.warn(`WebMCP: could not register ${tool.name}`, e)
    })
  }
  return true
}

/** Offers the game tools while the component that calls this is on screen. */
export function useWebMcpTools(game: GameActions): void {
  const controller = new AbortController()
  onMounted(() => {
    registerGameTools(game, controller.signal)
  })
  onBeforeUnmount(() => {
    controller.abort()
  })
}
