<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Phpml\Classification\SVC;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\Dataset\FilesDataset;
use Phpml\FeatureExtraction\StopWords\English;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Metric\Accuracy;
use Phpml\ModelManager;
use Phpml\Pipeline;
use Phpml\SupportVectorMachine\Kernel;
use Phpml\Tokenization\NGramTokenizer;
use Phpml\Tokenization\WordTokenizer;

/**
 * Class TextDataClassification
 * @package App\Console\Commands
 * @refrer https://arkadiuszkondas.com/text-data-classification-with-bbc-news-article-dataset/
 */
class TextDataClassification extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'whos:text-data-classification {--text=} {--renew}';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Text data classification with BBC news article dataset';

    /**
     * Create a new command instance.
     *
     * @return void
     */
    public function __construct()
    {
        parent::__construct();
    }

    /**
     * Execute the console command.
     *
     * @return mixed
     */
    public function handle()
    {
        $modelFile = storage_path('app/bbc.phpml');
        if ($this->option('renew') || ! file_exists($modelFile)) {
            $dataset = new FilesDataset(storage_path('app/bbc'));

            $split = new StratifiedRandomSplit($dataset, 0.01);

            $pipeline = new Pipeline([
                new TokenCountVectorizer(new NGramTokenizer(1, 3), new English()),
                new TfIdfTransformer()
            ], new SVC(Kernel::LINEAR));

            $pipeline->train($split->getTrainSamples(), $split->getTrainLabels());

            $predicted = $pipeline->predict($split->getTestSamples());
            $this->info('Accuracy: '.Accuracy::score($split->getTestLabels(), $predicted));

            (new ModelManager())->saveToFile($pipeline, $modelFile);
        } else {
            $pipeline = (new ModelManager())->restoreFromFile($modelFile);
        }

        if ($this->option('text')) {
            dump($pipeline->predict([$this->option('text')]));
        }
    }
}
