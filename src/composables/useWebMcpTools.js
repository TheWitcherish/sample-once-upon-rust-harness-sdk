import { onBeforeUnmount, onMounted } from 'vue';
/** Chrome's tool guidance caps one result at 1.5K characters. */
export const MAX_RESULT_CHARS = 1500;
function isRecord(value) {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
/** Chrome before 155 may hand the arguments over as a JSON string, later versions as an object. */
function toArgs(input) {
    if (typeof input === 'string') {
        try {
            const parsed = JSON.parse(input);
            return isRecord(parsed) ? parsed : {};
        }
        catch {
            return {};
        }
    }
    return isRecord(input) ? input : {};
}
function clip(text, max) {
    return text.length <= max ? text : `${text.slice(0, Math.max(0, max - 3))}...`;
}
/** The narration first, then the dice and the suggested moves, which always fit. */
export function formatStory(story) {
    const tail = [];
    const rolls = story.dice_rolls ?? [];
    if (rolls.length > 0) {
        tail.push(`Dice: ${rolls.map((r) => `${r.dice_type}=${r.result} (${r.reason})`).join(', ')}`);
    }
    const moves = story.actions_suggestions ?? [];
    if (moves.length > 0) {
        tail.push(`Suggested next moves: ${moves.join(' | ')}`);
    }
    const tailText = clip(tail.join('\n\n'), MAX_RESULT_CHARS / 2);
    const room = MAX_RESULT_CHARS - (tailText ? tailText.length + 2 : 0);
    const narration = clip((story.response ?? '').trim(), room);
    return tailText ? `${narration}\n\n${tailText}` : narration;
}
function gameTools(game) {
    return [
        {
            name: 'take_action',
            description: "Play the player's next turn in this adventure: one thing they say or do, in their own words, " +
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
                const action = toArgs(input).action;
                if (typeof action !== 'string' || action.trim() === '') {
                    return "Pass the player's move as a non-empty string in \"action\", for example \"I search the tavern for a quest board\".";
                }
                if (game.isBusy()) {
                    return 'The Game Master is still answering the last turn. Wait a moment, then try again.';
                }
                const story = await game.takeAction(action.trim());
                if (story === null) {
                    return `The Game Master could not answer (${game.lastError() ?? 'unknown error'}). Check that the Chapter 5 servers are running, then try again.`;
                }
                return formatStory(story);
            },
        },
        {
            name: 'read_character_sheet',
            description: "Read the player's character sheet: name, race, class, level, experience, ability scores and inventory.",
            inputSchema: { type: 'object', properties: {} },
            annotations: { readOnlyHint: true, untrustedContentHint: true },
            async execute() {
                try {
                    return clip(JSON.stringify(await game.readCharacterSheet()), MAX_RESULT_CHARS);
                }
                catch (e) {
                    const reason = e instanceof Error ? e.message : 'unknown error';
                    return `Could not read the character sheet (${reason}). Check that the Game Master is running.`;
                }
            },
        },
    ];
}
/** Registers the game tools until `signal` aborts. Returns false when the browser has no WebMCP. */
export function registerGameTools(game, signal) {
    const modelContext = document.modelContext;
    if (!modelContext) {
        return false;
    }
    for (const tool of gameTools(game)) {
        modelContext.registerTool(tool, { signal }).catch((e) => {
            console.warn(`WebMCP: could not register ${tool.name}`, e);
        });
    }
    return true;
}
/** Offers the game tools while the component that calls this is on screen. */
export function useWebMcpTools(game) {
    const controller = new AbortController();
    onMounted(() => {
        registerGameTools(game, controller.signal);
    });
    onBeforeUnmount(() => {
        controller.abort();
    });
}
