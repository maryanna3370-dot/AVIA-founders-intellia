const express = require('express');
const path = require('path');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;
const apiKey = process.env.GEMINI_API_KEY;

app.use(express.json());
app.use(express.static(path.join(__dirname, '/')));

function buildPrompt(problem, stakeholders) {
    let content = `Problem:\n${problem}\n\nStakeholder Feedback:\n`;
    if (!stakeholders || stakeholders.length === 0) {
        content += 'No stakeholder feedback found.';
    } else {
        stakeholders.forEach((stakeholder, index) => {
            content += `\nStakeholder ${index + 1}:\n`;
            content += `- Does this problem resonate with them? ${stakeholder.resonate}\n`;
            content += `- What aspects do they think matter most? ${stakeholder.aspects}\n`;
            content += `- What questions or concerns come to mind? ${stakeholder.questions}\n`;
            content += `- What are you missing about this problem? ${stakeholder.missing}\n`;
        });
    }
    return content;
}

function extractAnalysisText(data) {
    if (!data) return '';
    if (typeof data === 'string') return data;
    const candidate = data.candidates?.[0] ?? data?.output?.candidates?.[0];
    if (!candidate) {
        if (data.error) return JSON.stringify(data.error);
        return JSON.stringify(data);
    }
    if (typeof candidate === 'string') return candidate;
    if (candidate.text) return candidate.text;
    if (candidate.content && Array.isArray(candidate.content)) {
        const outputText = candidate.content.find(item => item.type === 'output_text');
        if (outputText?.text) return outputText.text;
        return candidate.content.map(item => item.text || JSON.stringify(item)).join(' ');
    }
    return JSON.stringify(candidate);
}

app.post('/api/analyze', async (req, res) => {
    if (!apiKey) {
        return res.status(500).json({ error: 'Server missing GEMINI_API_KEY in environment.' });
    }

    const { problem, stakeholders } = req.body;
    if (!problem || typeof problem !== 'string' || !problem.trim()) {
        return res.status(400).json({ error: 'Problem is required.' });
    }

    const promptText = buildPrompt(problem.trim(), Array.isArray(stakeholders) ? stakeholders : []);

    try {
        const response = await fetch('https://gemini.googleapis.com/v1/models/gemini-2.5-flash:generateText', {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                temperature: 0.7,
                maxOutputTokens: 1024,
                candidateCount: 1,
                prompt: {
                    messages: [
                        {
                            role: 'system',
                            content: 'You are an expert business analyst. Analyze the stakeholder feedback provided as it relates to the problem and structure your response with the following sections:\n\nPATTERNS: What themes appear across multiple sources? (3-5 bullets)\nSURPRISES: Where sources disagree and what that might mean (3-5 bullets)\nASSUMPTIONS TO REVISIT: What assumptions were challenged or proven wrong? (3-5 bullets)\nRED FLAGS: What concerns or risks surfaced? (3-5 bullets)\nTOP 3 INSIGHTS: The most important takeaways for your problem (3-5 bullets)',
                        },
                        {
                            role: 'user',
                            content: promptText,
                        },
                    ],
                },
            }),
        });

        const data = await response.json();
        if (!response.ok) {
            return res.status(response.status).json({ error: data?.error || 'Gemini request failed', details: data });
        }

        const analysis = extractAnalysisText(data);
        return res.json({ analysis, raw: data });
    } catch (error) {
        console.error('Gemini API error:', error);
        return res.status(500).json({ error: error.message || 'Unexpected server error.' });
    }
});

app.listen(port, () => {
    console.log(`Server listening on http://localhost:${port}`);
});
