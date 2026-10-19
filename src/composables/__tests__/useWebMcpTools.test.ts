import { afterEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h } from 'vue'
import { mount } from '@vue/test-utils'
import {
  MAX_RESULT_CHARS,
  formatStory,
  registerGameTools,
  useWebMcpTools,
  type GameActions,
} from '../useWebMcpTools'
import type { CharacterStats, StoryOutput } from '../../types'

interface RegisteredTool {
  name: string
  description: string
  inputSchema: Record<string, unknown>
  annotations?: Record<string, boolean>
  execute: (input: unknown) => Promise<string>
}

const STORY: StoryOutput = {
  response: 'The tavern falls silent as you walk in.',
  actions_suggestions: ['Order an ale', 'Read the quest board'],
  details: 'none',
  dice_rolls: [{ dice_type: 'd20', result: 14, reason: 'Perception' }],
}

const SHEET: CharacterStats = {
  character_id: '1',
  name: 'Thorin',
  character_class: 'Fighter',
  race: 'Dwarf',
  gender: 'Male',
  level: 1,
  experience: 0,
  stats: { strength: 16, dexterity: 12, constitution: 15, intelligence: 10, wisdom: 10, charisma: 8 },
  inventory: [{ item_name: 'Battleaxe', quantity: 1 }],
  created_at: '2026-10-19T00:00:00Z',
}

function installModelContext() {
  const tools: RegisteredTool[] = []
  const signals: AbortSignal[] = []
  const registerTool = vi.fn((tool: RegisteredTool, options: { signal: AbortSignal }) => {
    tools.push(tool)
    signals.push(options.signal)
    return Promise.resolve()
  })
  Object.defineProperty(document, 'modelContext', { value: { registerTool }, configurable: true })
  return { tools, signals, registerTool }
}

function fakeGame(overrides: Partial<GameActions> = {}): GameActions {
  return {
    takeAction: vi.fn().mockResolvedValue(STORY),
    readCharacterSheet: vi.fn().mockResolvedValue(SHEET),
    isBusy: () => false,
    lastError: () => null,
    ...overrides,
  }
}

function tool(tools: RegisteredTool[], name: string): RegisteredTool {
  const found = tools.find((t) => t.name === name)
  if (!found) throw new Error(`${name} was not registered`)
  return found
}

afterEach(() => {
  delete (document as { modelContext?: unknown }).modelContext
})

describe('registerGameTools', () => {
  it('does nothing in a browser without WebMCP', () => {
    expect(registerGameTools(fakeGame(), new AbortController().signal)).toBe(false)
  })

  it('registers take_action and read_character_sheet with the given signal', () => {
    const { tools, signals } = installModelContext()
    const controller = new AbortController()

    expect(registerGameTools(fakeGame(), controller.signal)).toBe(true)

    expect(tools.map((t) => t.name)).toEqual(['take_action', 'read_character_sheet'])
    expect(signals.every((s) => s === controller.signal)).toBe(true)
    expect(tool(tools, 'read_character_sheet').annotations).toEqual({
      readOnlyHint: true,
      untrustedContentHint: true,
    })
    expect(tool(tools, 'take_action').annotations?.readOnlyHint).toBeUndefined()
  })

  it('keeps names and descriptions within Chrome tool limits', () => {
    const { tools } = installModelContext()
    registerGameTools(fakeGame(), new AbortController().signal)

    for (const t of tools) {
      expect(t.name.length).toBeLessThanOrEqual(30)
      expect(t.description.length).toBeLessThanOrEqual(500)
    }
  })
})

describe('take_action', () => {
  it('plays the turn and returns the narration, dice and suggested moves', async () => {
    const { tools } = installModelContext()
    const game = fakeGame()
    registerGameTools(game, new AbortController().signal)

    const result = await tool(tools, 'take_action').execute({ action: '  I walk in  ' })

    expect(game.takeAction).toHaveBeenCalledWith('I walk in')
    expect(result).toContain('The tavern falls silent')
    expect(result).toContain('d20=14 (Perception)')
    expect(result).toContain('Order an ale | Read the quest board')
  })

  it('accepts its arguments as a JSON string', async () => {
    const { tools } = installModelContext()
    const game = fakeGame()
    registerGameTools(game, new AbortController().signal)

    await tool(tools, 'take_action').execute('{"action": "I order an ale"}')

    expect(game.takeAction).toHaveBeenCalledWith('I order an ale')
  })

  it('asks for an action instead of sending an empty turn', async () => {
    const { tools } = installModelContext()
    const game = fakeGame()
    registerGameTools(game, new AbortController().signal)

    const result = await tool(tools, 'take_action').execute({ action: '   ' })

    expect(game.takeAction).not.toHaveBeenCalled()
    expect(result).toContain('non-empty string in "action"')
  })

  it('waits its turn while another turn is in flight', async () => {
    const { tools } = installModelContext()
    const game = fakeGame({ isBusy: () => true })
    registerGameTools(game, new AbortController().signal)

    const result = await tool(tools, 'take_action').execute({ action: 'I attack' })

    expect(game.takeAction).not.toHaveBeenCalled()
    expect(result).toContain('still answering')
  })

  it('reports a failed turn with the error the page shows', async () => {
    const { tools } = installModelContext()
    const game = fakeGame({
      takeAction: vi.fn().mockResolvedValue(null),
      lastError: () => 'Request failed with status 500',
    })
    registerGameTools(game, new AbortController().signal)

    const result = await tool(tools, 'take_action').execute({ action: 'I attack' })

    expect(result).toContain('Request failed with status 500')
    expect(result).toContain('Chapter 5 servers')
  })
})

describe('read_character_sheet', () => {
  it('returns the sheet as JSON', async () => {
    const { tools } = installModelContext()
    registerGameTools(fakeGame(), new AbortController().signal)

    const result = await tool(tools, 'read_character_sheet').execute({})

    expect(JSON.parse(result)).toEqual(SHEET)
  })

  it('explains what went wrong when the sheet cannot be read', async () => {
    const { tools } = installModelContext()
    const game = fakeGame({ readCharacterSheet: vi.fn().mockRejectedValue(new Error('Request failed with status 404')) })
    registerGameTools(game, new AbortController().signal)

    const result = await tool(tools, 'read_character_sheet').execute({})

    expect(result).toContain('Request failed with status 404')
  })
})

describe('formatStory', () => {
  it('stays within the result budget and keeps the suggested moves', () => {
    const long = { ...STORY, response: 'A very long scene. '.repeat(200) }

    const result = formatStory(long)

    expect(result.length).toBeLessThanOrEqual(MAX_RESULT_CHARS)
    expect(result).toContain('Suggested next moves: Order an ale | Read the quest board')
    expect(result).toContain('...')
  })

  it('returns the narration alone when there are no dice or suggestions', () => {
    expect(formatStory({ ...STORY, dice_rolls: [], actions_suggestions: [] })).toBe(STORY.response)
  })
})

describe('useWebMcpTools', () => {
  it('offers the tools while mounted and withdraws them on unmount', () => {
    const { tools, signals } = installModelContext()
    const wrapper = mount(
      defineComponent({
        setup() {
          useWebMcpTools(fakeGame())
          return () => h('div')
        },
      }),
    )

    expect(tools).toHaveLength(2)
    expect(signals[0].aborted).toBe(false)

    wrapper.unmount()

    expect(signals.every((s) => s.aborted)).toBe(true)
  })
})
